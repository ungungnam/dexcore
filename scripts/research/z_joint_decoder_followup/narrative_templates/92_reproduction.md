**Outputs** (all under `/result/uhnam/dexcore/reports/z_joint_decoder_followup/`): `report.md`, `dashboard.html`, `experiment_config.json`
(models, recipe, decision rule, checkpoint paths and md5), `model_table.csv`, `training_summary.csv`, `main_metrics.csv`,
`structural_metrics.csv`, `wrench_metrics.csv`, `temporal_stability.csv`, `z_metrics.csv`, `decoder_diagnostic.csv`, `horizon_metrics.csv`,
`paired_comparisons.csv`, `decision_summary.csv`, `heldout_probe.csv`, `sanity_summary.json`, `report_values.json`, `temporal_curves.npz`,
`figures/`, `logs/` (training, queue, post-processing, probes), `narrative/`, `scripts/` (a copy of the code with `MD5SUMS.txt`), and per
dataset `preds/`, `metrics/`, `diag/`, `probe/`, `train_logs/`.

**Checkpoints.**

{{ckpt_table}}

**Code**: `scripts/research/z_joint_decoder_followup/` in the repository (not committed).

**Exact reproduction**: `bash scripts/research/z_joint_decoder_followup/pipeline.sh`. Its steps:

```bash
cd /home/uhnam/workspace/dexcore/scripts/research/z_joint_decoder_followup
export PYTHONPATH=/home/uhnam/workspace/dexcore; PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
# training through the shared-GPU queue (or directly: CUDA_VISIBLE_DEVICES=5 $PY zj_train.py --dataset taco --model M2 --wandb --mem-cap-mib 9500)
printf 'taco M2\narctic M2\ntaco M3\narctic M3\ntaco M0r\narctic M0r\n' > /result/uhnam/dexcore/reports/z_joint_decoder_followup/logs/queue.txt
$PY zj_queue.py
# generation, metrics, re-check of the Stage-2 metric files, diagnostics, sanity checks
bash zj_post.sh 5 taco; bash zj_post.sh 5 arctic
# held-out probes (their trigger fired)
for ds in taco arctic; do CUDA_VISIBLE_DEVICES=5 $PY zj_probe.py --dataset $ds --model M2; CUDA_VISIBLE_DEVICES=5 $PY zj_probe.py --dataset $ds --model M2 --shrink; done
# tables, figures, report, dashboard
$PY zj_sanity.py --merge; $PY zj_aggregate.py; CUDA_VISIBLE_DEVICES="" $PY zj_figures.py; $PY zj_report.py; $PY zj_dashboard.py
```

Training logs are also on Weights & Biases, project `dexcore-z-joint-decoder`.
