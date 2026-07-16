"""Temporary neutral boundary around the proven temporal-caption data pipeline.

This module re-exports only dataset/collation contracts. It must not construct
or score a v1/v2 model, and may be replaced by a package-level data module
without changing QCPR v3 architecture.
"""
from __future__ import annotations

import ucv2_stage1_next_core as _data

Stage1NextConfig = _data.Stage1NextConfig
build_datasets = _data._build_stage1_datasets
assert_disjoint = _data._assert_stage1_disjoint
caption_frequencies = _data.caption_frequencies
make_collator = _data._make_collator
make_train_loader = _data.make_train_loader
make_eval_loader = _data.make_eval_loader
retrieval_supervision_selection = _data.retrieval_supervision_selection

