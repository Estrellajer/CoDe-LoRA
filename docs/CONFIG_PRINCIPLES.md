# Configuration principles

One setting for every benchmark and method (`configs/protocols/*`, `configs/methods/*`): AdamW, one epoch per task,
effective batch 64, bf16, LoRA r = 8 / alpha = 32, generation evaluated in prompt-length order (`eval.sort_by_length:
true`; fp32 outputs do not change, bf16 may flip about 1-2 % of the examples because batch padding changes). Hyper-parameters
are not searched per method; a method's own constants (penalty weights, MoLE-CIE's expert count, ...) are the ones of its
original paper. The learning rate depends on the backbone and, for Qwen3-0.6B, on the method family; it is judged by the
final AP / BWT, not by the score on the task just trained.

| backbone | lr |
|---|---|
| T5-large | 1e-3 for every method, except plain sequential LoRA: 1e-4 (at 1e-3 it collapses on Long, AP 40.7 +- 13.6) |
| Llama-2-7B, Qwen3.5-4B | 1e-4 for every method (at 1e-3 the shared-adapter methods diverge, gradient norm 100-340) |
| Qwen3-0.6B | 1e-3 for the per-task-expert methods (De-LoRA, CoDe-LoRA); 1e-4 for the methods that train one shared adapter (LoRA, O-LoRA, N-LoRA, Co-LoRA, MoLE-CIE) |

Evidence for the Qwen3-0.6B split: at 1e-3 the shared-adapter methods forget catastrophically (LoRA on Long: AP 10.5, BWT
-62); at 1e-4 a new expert cannot learn the tasks that have only 4-6 updates per epoch (De-LoRA: AP 71.2 to 59.5).

Method-specific settings that are not learning rates: O-LoRA uses lambda_l2 = 0 (the official value); MoLE-CIE runs its
forward in 16 x 4 micro-batches for the same effective batch; the large-vocabulary decoders score candidate labels in
smaller batches (`eval.score_batch_size`: Qwen3-0.6B 32, Qwen3.5-4B 8, because the loss upcasts batch x length x vocabulary
logits).

CoDe-LoRA adds no tuned constants of its own: the retraction rank is `lora.r`, the experts are plain LoRA adapters
(dropout 0) with the backbone's lr, the fold uses the fixed `sqrt` coefficients with the null-space projection
(`colora.scaling`, `colora.projection`; ablations `additive` and `none`), and the PMI fusion (on in `methods/code-lora`,
off with `evaluation/no-fusion`; `fusion.stages: final` fuses only the last evaluation stage) selects its single weight per task on dev from the fixed grid `{0, 0.1, 0.25, 0.5, 1}`.
