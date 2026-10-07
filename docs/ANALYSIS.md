# Analysis: tables and figures from run directories

`pip install -e ".[analysis]"` (matplotlib; tables need nothing extra). Inputs are run directories as described in
[RESULTS.md](RESULTS.md); any directory containing `summary.json` below the given paths is a run. Outputs go to `--out`
(figures as PNG and PDF, tables as `.md` / `.tex` / `.csv`).

```bash
R=outputs/table1                                    # root of many runs (seeds x methods x orders)
python -m codelora.analysis table       $R --out figs --format latex --methods lora o-lora n-lora co-lora de-lora code-lora mole-cie
python -m codelora.analysis compare     $R --out figs --reference code-lora      # deltas to a reference method
python -m codelora.analysis efficiency  $R --out figs                            # time, memory, parameters
python -m codelora.analysis curves      $R --out figs      # AP vs number of tasks (needs eval.matrix=full)
python -m codelora.analysis forgetting  $R --out figs      # per task: peak vs final, mean forgetting in the title
python -m codelora.analysis routing     $R --out figs      # gold task -> routed task, from results.json
python -m codelora.analysis fwt         $R --out figs      # needs eval.forward_transfer (and eval.matrix=full)
python -m codelora.analysis all         $R --out figs      # everything the directory supports; skipped commands say why
```

* `table` groups runs by (backbone, task order, method) and reports mean ± std over seeds of AP and BWT in percent; the best
  value per column and block is bold. `compare` adds the difference to `--reference`.
* `curves`: one figure per (backbone, order), one line per method. Runs stored with `diagonal_final` have no intermediate AP;
  they are drawn as the final AP at the last task and the title says so. Use `eval.matrix=full` for the curves.
* `forgetting`: *peak* is the best score the task had in the stored matrix after it was learned; with `diagonal_final` that is
  the score right after learning. Forgetting = peak - final.
