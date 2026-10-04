**Outputs** (all under `/result/uhnam/dexcore/reports/z_stateful_s0_factorial/`): `report.md`, `dashboard.html`, `experiment_config.json`
(models, recipe, decision rule, checkpoint paths and md5), `model_table.csv`, `taco_metrics.csv`, `arctic_metrics.csv`,
`latent_metrics.csv`, `horizon_metrics.csv`, `early_frame_metrics.csv`, `factorial_effects.csv`, `direct_comparison.csv`,
`decision_summary.csv`, `paired_comparisons.csv`, `training_summary.csv`, `sanity_summary.json`, `report_values.json`,
`temporal_curves.npz`, `init_aug_decision.json`, `figures/`, `logs/` (training, queue, pilots, post-processing, discarded attempts),
`narrative/`, `scripts/` (a copy of the code with `MD5SUMS.txt`), and per dataset `preds/`, `metrics/`, `diag/`, `train_logs/`.

**Checkpoints.**

{{ckpt_table}}

M00 and D0 are reused files of the two previous studies and were not retrained. The two discarded M10 checkpoints of the first
launch are in `/ckpt/uhnam/dexcore/z_stateful_s0_factorial/discarded_shared_init/` and are used nowhere.

**Code**: `scripts/research/z_stateful_s0_factorial/` in the repository (not committed).

**Exact reproduction**: `bash scripts/research/z_stateful_s0_factorial/pipeline.sh`. Its steps:

```bash
cd /home/uhnam/workspace/dexcore/scripts/research/z_stateful_s0_factorial
export PYTHONPATH=/home/uhnam/workspace/dexcore; PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
O=/result/uhnam/dexcore/reports/z_stateful_s0_factorial
# 1 pilot for the initial-state supervision (validation data only), then the rule
for ds in taco arctic; do for k in 0 3; do CUDA_VISIBLE_DEVICES=4 $PY zf_pilot_init.py --dataset $ds --aug $k --steps 2000; done; done
$PY zf_pilot_decide.py --step 2000
# 2 training through the shared-GPU queue, one seed
#   (or directly: CUDA_VISIBLE_DEVICES=4 $PY zf_train.py --dataset taco --model M10 --seed 0 --wandb --mem-cap-mib 34000 --no-dec-ckpt)
printf 'taco M10\ntaco M11\narctic M10\narctic M11\ntaco M01\narctic M01\n' > $O/logs/queue.txt
$PY zf_queue.py
# 3 generation, metrics on test and validation, latent diagnostics, sanity checks
bash zf_post.sh 4 taco; bash zf_post.sh 4 arctic
# 4 tables, figures, report, dashboard
$PY zf_sanity.py --merge; $PY zf_aggregate.py; CUDA_VISIBLE_DEVICES="" $PY zf_figures.py; $PY zf_report.py; $PY zf_dashboard.py
```

Training logs are also on Weights & Biases, project `dexcore-z-stateful-s0-factorial`. The stateful runs reported here have the
run-id suffix `_indep`.
