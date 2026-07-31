import csv
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
project = Path('/mnt/weka/svardanyan/rs_change_project')
r1 = project / 'runs/qcpr_retrieval_repair_r1_20260727-165844/r1'
(root / 'reports').mkdir(parents=True, exist_ok=True)
(root / 'plots').mkdir(parents=True, exist_ok=True)

jobs = []
for row in csv.DictReader((root / 'jobs.tsv').open(), delimiter='|'):
    jid = row.get('JobID', '')
    if not jid or '.' in jid:
        continue
    name = row.get('JobName', '').strip()
    low = name.lower()
    if jid in {'199522', '199527', '200097'} or 'repair-r1' in low:
        lineage = 'R1_repair'
    elif any(x in low for x in ('retrieval-full', 'query-grounding', 'qcpr-evaluation', 'qcpr-report')):
        lineage = 'single_pass_chain'
    elif 'dataset' in low or 'source-probe' in low:
        lineage = 'dataset_v2'
    elif any(x in low for x in ('a0', 'm0', 'grounding')):
        lineage = 'earlier_a0_m0'
    else:
        lineage = 'other'
    jobs.append({'run_id': f'slurm:{jid}', 'slurm_job_id': jid, 'job_name': name,
                 'state': row.get('State'), 'exit_code': row.get('ExitCode'),
                 'elapsed': row.get('Elapsed'), 'max_rss': row.get('MaxRSS'),
                 'start': row.get('Start'), 'end': row.get('End'), 'lineage': lineage,
                 'scientific_failure': False if jid == '199527' else None})

cfg = json.loads((r1 / 'run_config.json').read_text())
acc = json.loads((r1 / 'r1_acceptance.json').read_text())
batch = json.loads((r1 / 'batch_contract.json').read_text())
for row in jobs:
    if row['slurm_job_id'] == '199527':
        row['classification'] = 'infrastructure_only_failure'
    if row['slurm_job_id'] == '200097':
        row.update({'code_sha': cfg['git_sha'], 'manifest_sha': cfg['validation_manifest_sha256'],
                    'checkpoint_sha256': acc['checkpoint_sha256'],
                    'initial_checkpoint_sha256': cfg['baseline_checkpoint_sha256'],
                    'architecture': 'residual dense-token adapters + 3 PAIR cross-attention blocks; frozen Jina/UniverSat',
                    'trainable_parameters': 'not recorded in R1 metadata', 'optimizer': 'AdamW',
                    'scheduler': 'cosine with warmup', 'logical_batch': 128, 'microbatch': 16,
                    'logical_score_matrix': '256x128', 'steps': 1044, 'epochs': 12,
                    'hard_negative_policy': 'warmup 3 epochs; 32 model-mined candidates; refresh every 2 epochs',
                    'peak_vram': f"{batch['peak_allocated_gib']} GiB allocated / {batch['peak_reserved_gib']} GiB reserved",
                    'runtime': '01:51:20', 'metric_protocol': 'QCPR_UNIFIED exact physical-pair development'})
jobs += [{'run_id': 'checkpoint:epoch19-baseline', 'lineage': 'R0_baseline', 'state': 'COMPLETED',
          'checkpoint_sha256': 'd3861156a0ec9dde81e56f6a8d3d1c6e1e49df301b5d0125338bee7d75ea9ac6',
          'classification': 'accepted_historical_baseline'},
         {'run_id': 'checkpoint:r1-best-epoch9', 'slurm_job_id': '200097', 'lineage': 'R1_repair',
          'state': 'COMPLETED', 'checkpoint_sha256': acc['checkpoint_sha256'], 'classification': 'best_R1_epoch9'}]
(root / 'reports/historical_run_registry.jsonl').write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in jobs))
lines = ['# Historical QCPR run registry', '', 'Job 199527 is infrastructure-only, not a scientific failure.', '', '| Job | Name | State | Exit | Lineage |', '|---:|---|---|---|---|']
lines += [f"| {x.get('slurm_job_id','')} | {x.get('job_name','')} | {x.get('state','')} | {x.get('exit_code','')} | {x.get('lineage','')} |" for x in jobs if x.get('slurm_job_id')]
(root / 'reports/historical_run_registry.md').write_text('\n'.join(lines) + '\n')

metrics = [json.loads(x) for x in (r1 / 'metrics.jsonl').read_text().splitlines()]
fields = []
for d in metrics:
    x, c, n, o = d['development']['all'], d['development']['changed'], d['development']['no_change'], d['ordering']
    fields.append({'epoch': d['epoch'], 'global_step': d['global_step'], 'loss': d['loss'],
                   'positive_loss': d['positive_loss'], 'negative_loss': d['negative_loss'],
                   'positive_similarity': d['positive_similarity'], 'negative_similarity': d['negative_similarity'],
                   'mrr': x['mrr'], 'r1': x['recall_at_1'], 'r5': x['recall_at_5'], 'r10': x['recall_at_10'],
                   'mean_rank': x['mean_rank'], 'median_rank': x['median_rank'], 'changed_mrr': c['mrr'],
                   'changed_r10': c['recall_at_10'], 'no_change_mrr': n['mrr'], 'no_change_r10': n['recall_at_10'],
                   'hard_refreshes': d['hard_negative_refreshes'], 'hard_insertions': o['hard_neighbour_insertions']})
with (root / 'reports/r1_epoch_metrics.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(fields[0])); w.writeheader(); w.writerows(fields)
checkpoints = []
for p in sorted(r1.glob('*.pt')):
    checkpoints.append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                        'size_bytes': p.stat().st_size, 'role': 'best' if p.name.startswith('best') else 'final'})
(root / 'reports/r1_checkpoint_inventory.json').write_text(json.dumps({'checkpoints': checkpoints, 'per_epoch_checkpoint_files': [], 'note': 'Trajectory is from metrics.jsonl; only best/final checkpoints were persisted.'}, indent=2) + '\n')
refresh = [{'epoch': d['epoch'], 'refresh_count': d['hard_negative_refreshes'], 'insertions': d['ordering']['hard_neighbour_insertions'],
            'loss': d['loss'], 'mrr': d['development']['all']['mrr'], 'r10': d['development']['all']['recall_at_10']}
           for d in metrics if d['hard_negative_refreshes']]
(root / 'reports/r1_hard_negative_analysis.json').write_text(json.dumps({'warmup_epochs': 3, 'refresh_effect': refresh, 'false_negative_policy': 'ambiguity-aware exclusions from manifest', 'interpretation': 'The first hard refresh coincides with a loss jump and exact-metric regression; later refreshes recover partially.'}, indent=2) + '\n')
best = max(fields, key=lambda x: x['mrr']); base = acc['baseline_metrics']['all']
(root / 'reports/r1_trajectory_summary.md').write_text('# R1 trajectory summary\n\n' +
    f"- Baseline MRR: {base['mrr']:.6f}; R@10: {base['recall_at_10']:.6f}; median rank: {base['median_rank']:.1f}.\n" +
    f"- Best R1 epoch: {best['epoch']} (MRR {best['mrr']:.6f}, R@10 {best['r10']:.6f}, median {best['median_rank']:.1f}).\n" +
    '- Gate: failed (MRR >= 0.030, R@10 >= 0.060, median <= 150).\n' +
    '- Pre-mining epochs: 1-3. Epoch 4 is the first hard-refresh epoch.\n' +
    '- Job 199527 is infrastructure-only and is not a scientific failure.\n')
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    e = [x['epoch'] for x in fields]
    def plot(name, ys, labels, ylabel, marker=False):
        plt.figure(figsize=(9, 5))
        for y, label in zip(ys, labels): plt.plot(e, y, marker='o', label=label)
        if marker: plt.axvline(3.5, color='red', linestyle='--', label='first hard refresh')
        plt.xlabel('epoch'); plt.ylabel(ylabel); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(root / 'plots' / name, dpi=160); plt.close()
    plot('r1_exact_trajectory.png', [[x['mrr'] for x in fields], [x['r10'] for x in fields]], ['MRR', 'Recall@10'], 'score', True)
    plot('r1_changed_nochange.png', [[x['changed_mrr'] for x in fields], [x['no_change_mrr'] for x in fields]], ['changed MRR', 'no-change MRR'], 'MRR')
    plot('r1_hard_refresh_effect.png', [[x['loss'] for x in fields], [x['hard_insertions'] for x in fields]], ['loss', 'hard insertions'], 'value', True)
except Exception as exc:
    (root / 'plots/plot_error.txt').write_text(repr(exc))
print(json.dumps({'root': str(root), 'epochs': len(fields), 'best_epoch': best['epoch'], 'best_mrr': best['mrr'], 'baseline_mrr': base['mrr']}, indent=2))
