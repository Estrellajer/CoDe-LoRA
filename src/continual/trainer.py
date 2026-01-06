import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Union, List, Tuple
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import (
    is_sagemaker_mp_enabled, 
    IntervalStrategy, 
    TrainerState, 
    TrainerControl,
    TrainingArguments,
    EvalLoopOutput,
    has_length,
    find_batch_size,
    nested_numpify,
    nested_concat,
    nested_truncate,
    denumpify_detensorize,
    is_deepspeed_zero3_enabled,
    deepspeed_init
)
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import DataLoader, IterableDataset

# Internal imports from the workspace
from .utils import skip_instructions, check_model, SUPPORTED_DECODER_MODELS


class DenserEvalCallback(TrainerCallback):
    """Callback to evaluate more frequently at the start of training."""
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        log_eval_steps = [1, 50, 100, 200]
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True
        return control


class ContinualTrainer(Seq2SeqTrainer):
    """
    A unified trainer for continual learning.
    Integrates UIETrainer logic and provides hooks for specific CL strategies.
    """
    
    def __init__(self, *args, model_strategy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_strategy = model_strategy

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        # Print parameter statistics once per training run (rank 0 only).
        # This is useful to verify LoRA-only training and track trainable parameters.
        if not hasattr(self, "_printed_param_stats"):
            self._printed_param_stats = True
            try:
                is_main = False
                # Prefer Trainer's built-in helper if available
                if hasattr(self, "is_world_process_zero") and callable(getattr(self, "is_world_process_zero")):
                    is_main = bool(self.is_world_process_zero())
                else:
                    # Best-effort env-based fallback
                    import os

                    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_PROCID") or "0"
                    is_main = int(rank) == 0
            except Exception:
                is_main = True

            if is_main:
                m = model.module if hasattr(model, "module") else model
                total = sum(p.numel() for p in m.parameters())
                trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
                lora_total = sum(
                    p.numel()
                    for n, p in m.named_parameters()
                    if ("lora_" in n or "loranew_" in n)
                )
                lora_trainable = sum(
                    p.numel()
                    for n, p in m.named_parameters()
                    if ("lora_" in n or "loranew_" in n) and p.requires_grad
                )
                pct = (100.0 * trainable / total) if total else 0.0
                print(
                    f"[Params] total={total:,} trainable={trainable:,} ({pct:.4f}%) "
                    f"lora_total={lora_total:,} lora_trainable={lora_trainable:,}"
                )

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        # Hook for specific CL loss (e.g. Orthogonal, EWC, etc.)
        if hasattr(self.model_strategy, "compute_additional_loss"):
            add_loss = self.model_strategy.compute_additional_loss(model, inputs, loss)
            loss = loss + add_loss

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            from transformers.utils import is_apex_available
            if is_apex_available():
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
        elif self.deepspeed:
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Prediction/evaluation loop with support for generation and custom metrics."""
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        if args.deepspeed and not self.deepspeed:
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size
        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = dataloader.dataset

        # Initialize containers
        losses_host, preds_host, labels_host = None, None, None
        all_losses, all_preds, all_labels = None, None, None
        observed_num_examples = 0

        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                if batch_size is None: batch_size = observed_batch_size

            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    all_losses = nested_numpify(losses_host) if all_losses is None else np.concatenate((all_losses, nested_numpify(losses_host)), axis=0)
                if preds_host is not None:
                    all_preds = nested_numpify(preds_host) if all_preds is None else nested_concat(all_preds, nested_numpify(preds_host), padding_index=-100)
                if labels_host is not None:
                    all_labels = nested_numpify(labels_host) if all_labels is None else nested_concat(all_labels, nested_numpify(labels_host), padding_index=-100)
                losses_host, preds_host, labels_host = None, None, None

        # Final gather
        if losses_host is not None:
            all_losses = nested_numpify(losses_host) if all_losses is None else np.concatenate((all_losses, nested_numpify(losses_host)), axis=0)
        if preds_host is not None:
            all_preds = nested_numpify(preds_host) if all_preds is None else nested_concat(all_preds, nested_numpify(preds_host), padding_index=-100)
        if labels_host is not None:
            all_labels = nested_numpify(labels_host) if all_labels is None else nested_concat(all_labels, nested_numpify(labels_host), padding_index=-100)

        num_samples = len(eval_dataset) if has_length(eval_dataset) else observed_num_examples
        if all_losses is not None: all_losses = all_losses[:num_samples]
        if all_preds is not None: all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None: all_labels = nested_truncate(all_labels, num_samples)

        metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix) if self.compute_metrics else {}
        metrics["global_step"] = self.state.global_step
        metrics = denumpify_detensorize(metrics)
        if all_losses is not None: metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform an evaluation step with generation support."""
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        # Prepare generation kwargs
        gen_kwargs = self._gen_kwargs.copy()
        gen_kwargs["synced_gpus"] = is_deepspeed_zero3_enabled()
        
        # Avoid TypeError: '>' not supported between instances of 'NoneType' and 'int'
        if gen_kwargs.get("num_beams") is None:
            gen_kwargs["num_beams"] = getattr(self.model.config, "num_beams", 1)
        if gen_kwargs["num_beams"] is None:
            gen_kwargs["num_beams"] = 1
            
        # Handle max_length vs max_new_tokens to avoid warnings
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is not None:
            gen_kwargs["max_new_tokens"] = gen_kwargs.pop("max_length")

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        # Filter out None values to let GenerationConfig use its defaults
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        generation_config = GenerationConfig(**gen_kwargs)
        main_input_name = self.model.main_input_name
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[main_input_name]

        generated_tokens = self.model.generate(input_ids=generation_inputs, generation_config=generation_config)

        # Padding logic for decoder models
        bs, source_len = inputs['input_ids'].shape
        is_decoder = check_model(self.model.config._name_or_path, SUPPORTED_DECODER_MODELS)
        max_length = (source_len + gen_kwargs["max_new_tokens"]) if is_decoder else gen_kwargs["max_new_tokens"]

        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"] if has_labels else None
        if labels is not None and labels.shape[-1] < gen_kwargs["max_new_tokens"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])

        return (loss, generated_tokens, labels)
