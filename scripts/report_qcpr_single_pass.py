from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument("--run-root",type=Path,required=True); a=p.parse_args()
 evaluation=json.loads((a.run_root/"evaluation/evaluation.json").read_text())
 retrieval=evaluation["retrieval"]["all"]; localization=evaluation["localization"]
 lines=["# QCPR single-pass retrieval and emergent localization","",
 "## 1. Did single-pass exact-pair retrieval improve?","",
 f"Final exact physical-pair Recall@1: {retrieval['recall_at_1']:.4f}; Recall@5: {retrieval['recall_at_5']:.4f}; Recall@10: {retrieval['recall_at_10']:.4f}. Compare against the preserved baseline table in the run contract; no segmentation metric selected this checkpoint.","",
 "## 2. Where does the true pair appear in Top-10?","",
 f"MRR: {retrieval['mrr']:.4f}; mean rank: {retrieval['mean_rank']:.2f}; median rank: {retrieval['median_rank']:.1f}; nDCG@10: {retrieval['ndcg_at_10']:.4f}. Exact per-query ranks are in evaluation/retrieval_top10.jsonl.","",
 "## 3. Is the query-conditioned map non-collapsed?","",
 f"Mean thresholded area ratio: {localization['map_area_ratio']:.4f}; mean map change across alternate captions: {localization['query_map_change']:.6f}.","",
 "## 4. Does the map change with query/pair?","",
 f"Mean absolute query-conditioned map change: {localization['query_map_change']:.6f}; pair-conditioned change: {localization['pair_map_change']:.6f}.","",
 "## 5. Held-out mask overlap (evaluation only)","",
 f"Soft Dice: {localization['soft_dice']:.4f}; soft IoU: {localization['soft_iou']:.4f}; pointing accuracy: {localization['pointing_accuracy']:.4f}; localization mass inside target: {localization['mass_inside']:.4f}.",
 "",
 "Masks were loaded only after retrieval and localization were frozen. They did not affect gradients, optimizers, early stopping, or checkpoint selection.",
 "",
 "## Architecture",
 "",
 "Frozen UniverSat emits aligned per-time grids at the configured native grid resolution. Per-cell temporal descriptors (T1, T2, signed delta, absolute delta, product) are projected to 384 dimensions, receive learned 2D positions, and are contextualized with CLS_PAIR by a six-layer Transformer. CLS_PAIR is projected to one normalized 512-D gallery vector. Frozen Jina pooled features are projected to one normalized 512-D query vector. Localization uses query-to-patch cross-attention and pair-level sigmoid supervision only."]
 (a.run_root/"report/report.md").write_text("\n".join(lines)+"\n")
 (a.run_root/"report/complete.json").write_text(json.dumps({"status":"complete","evaluation":str(a.run_root/"evaluation/evaluation.json")},indent=2)+"\n")
if __name__=="__main__": main()
