import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from .utils import check_model, SUPPORTED_DECODER_MODELS, SUPPORTED_SEQ2SEQ_MODELS

@dataclass
class ContinualDataCollator:
    """
    A self-contained data collator for continual learning.
    Supports both Seq2Seq and Decoder-only models.
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_dataset_name: bool = False
    common_dataset_name: str = None
    text_only: bool = False
    num_examples: int = 0
    input_record_file: str = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        model_name = getattr(self.model.config, "_name_or_path", "")
        
        if check_model(model_name, SUPPORTED_DECODER_MODELS):
            return self.decoder_call(batch)
        elif check_model(model_name, SUPPORTED_SEQ2SEQ_MODELS):
            return self.seq2seq_call(batch)
        else:
            raise ValueError(f"Unsupported model {model_name}!")

    def get_instruction(self, instance: Dict[str, Any]) -> str:
        """Construct the instruction string with optional task/dataset prefixes."""
        instruction = instance['Instance'].get("instruction", "")
        content = instance['Instance'].get('sentence', "")

        prefix = ''
        if self.add_task_name and 'Task' in instance:
            prefix += f"Task:{instance['Task']}\n"
        if self.add_dataset_name and 'Dataset' in instance:
            ds_name = self.common_dataset_name if self.common_dataset_name else instance['Dataset']
            prefix += f"Dataset:{ds_name}\n"
        
        if prefix:
            instruction = prefix + instruction

        # Handle few-shot samples if needed (placeholder for now)
        if instance.get('Samples') and len(instance['Samples']) > 0:
            # Logic for samples could be added here
            pass

        try:
            # Try to format the instruction with content if {} is present
            if "{}" in instruction:
                instruction = instruction.format(content)
            else:
                instruction = instruction + "\n" + content
        except Exception:
            # Fallback if formatting fails
            instruction = instruction + "\n" + content
            
        return instruction

    def seq2seq_call(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sources = []
        labels = []

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)
            
            source = instruction
            tokenized_source = self.tokenizer(
                source, 
                add_special_tokens=False, 
                truncation=True, 
                max_length=self.max_source_length
            )["input_ids"]
            sources.append(self.tokenizer.decode(tokenized_source, skip_special_tokens=True))

        if self.text_only:
            return {"inputs": sources, "labels": labels}

        model_inputs = self.tokenizer(
            sources,
            max_length=self.max_source_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        
        # `as_target_tokenizer` is deprecated in transformers v4 and will be removed in v5.
        # Use `text_target` to tokenize labels instead.
        labels_batch = self.tokenizer(
            text_target=labels,
            max_length=self.max_target_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        
        label_mask = labels_batch["attention_mask"].bool()
        model_inputs["labels"] = labels_batch["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            model_inputs["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])

        return model_inputs

    def decoder_call(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Causal LMs usually use left padding for batch generation
        self.tokenizer.padding_side = 'left'
        sources = []
        label_lens = []
        labels_text = []
        
        is_train = any(instance.get('subset') == 'train' for instance in batch)
        limit_input_len = (self.max_source_length + self.max_target_length) if is_train else self.max_source_length

        for instance in batch:
            label = instance['Instance']['label']
            labels_text.append(label)
            instruction = self.get_instruction(instance)

            # Add BOS and EOS if available
            bos = self.tokenizer.bos_token if self.tokenizer.bos_token else ""
            eos = self.tokenizer.eos_token if self.tokenizer.eos_token else ""
            
            task_input = bos + instruction
            full_label = label + eos

            tokenized_input = self.tokenizer(
                task_input, 
                add_special_tokens=False, 
                truncation=True, 
                max_length=limit_input_len
            )["input_ids"]
            tokenized_label = self.tokenizer(
                full_label, 
                add_special_tokens=False, 
                truncation=True, 
                max_length=self.max_target_length
            )["input_ids"]

            if instance.get('subset') in ['dev', 'test']:
                label_lens.append(0)
                if len(tokenized_input) <= limit_input_len:
                    sources.append(task_input)
                else:
                    sources.append(self.tokenizer.decode(tokenized_input[:limit_input_len], skip_special_tokens=False))
            else:
                # Training: combine input and label
                if len(tokenized_input) + len(tokenized_label) <= limit_input_len:
                    label_lens.append(len(tokenized_label))
                    sources.append(task_input + full_label)
                else:
                    # Truncate
                    combined = (tokenized_input + tokenized_label)[:limit_input_len]
                    sources.append(self.tokenizer.decode(combined, skip_special_tokens=False))
                    label_lens.append(max(0, limit_input_len - len(tokenized_input)))

        if self.text_only:
            return {"inputs": sources, 'labels': labels_text}

        model_inputs = self.tokenizer(
            sources,
            max_length=limit_input_len,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        attention_mask = model_inputs["attention_mask"].bool()
        model_inputs["labels"] = model_inputs['input_ids'].masked_fill(~attention_mask, self.label_pad_token_id)

        # Create loss mask: only compute loss on the label tokens
        # For decoder models, we usually mask out the instruction part in labels
        loss_mask = torch.ones_like(model_inputs['input_ids'], dtype=torch.float)
        
        for i, l_len in enumerate(label_lens):
            # The input is left-padded. The label is at the end.
            # Sequence: [PAD, PAD, ..., INST, LABEL]
            # Length of sequence is model_inputs['input_ids'].shape[1]
            seq_len = attention_mask[i].sum().item()
            if l_len > 0:
                # Mask out everything except the last l_len tokens
                # seq_len is the number of non-pad tokens
                # non-pad tokens start at shape[1] - seq_len
                # label tokens are the last l_len tokens
                label_start = model_inputs['input_ids'].shape[1] - l_len
                loss_mask[i, :label_start] = 0
            else:
                # For inference, we might not need a loss mask, but for consistency:
                loss_mask[i, :] = 0
        
        model_inputs['loss_mask'] = loss_mask.masked_fill(~attention_mask, 0)

        return model_inputs

