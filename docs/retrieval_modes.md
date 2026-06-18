# Retrieval Modes

Retrieval is not one task. The platform now treats the following as distinct modes:

## `static_region`

Single-patch similarity retrieval without explicit temporal change reasoning.

## `pair_analog`

Before/after pair retrieval for analogous transition events.

## `text_bitemporal`

Text-to-change retrieval for queries such as:

- `wetting`
- `new buildings`
- `cropland -> built_up`

## `novelty_single_image`

Single-image novelty or archive-comparison mode. This is not formal bi-temporal change detection.

## `transition_conditioned`

Retrieval constrained by an explicit semantic transition label or histogram direction such as:

- `cropland -> built_up`
- `vegetation -> bare_ground`
- `water -> dryland`

## `trajectory`

Short time-series retrieval for longer temporal behavior, not only T1/T2.

This is the right later-stage home for datasets such as `DynamicEarthNet` and `SpaceNet 7`.
