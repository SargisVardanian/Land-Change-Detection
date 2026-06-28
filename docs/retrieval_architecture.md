# Retrieval Architecture

## Core Position

Retrieval is a research-layer subsystem, not an extension of the locked demo explanation path.

The scientific target is also constrained:

- `LEVIR-CC` is the clean first text-to-pair benchmark
- `LEVIR-MCI` is the first grounded retrieval and segmentation benchmark
- `SECOND-CC` and `Hi-UCD` are the semantic transition bridge
- `DynamicEarthNet` and `SpaceNet 7` are later temporal-retrieval datasets, not the default first download

The architecture is intentionally additive:

- existing `app.py` normal mode remains unchanged
- retrieval lives under `src/land_change_detection/retrieval/`
- research orchestration lives under `src/land_change_detection/pipelines/research_pipeline.py`
- optional UI lives in a separate research page

## Online vs Offline Split

### Offline

- scene discovery
- manifest building
- patch pairing
- embedding generation
- vector index creation
- dataset curation

### Online

- accept query
- choose retrieval mode
- encode query
- search target index
- rerank with metadata
- return structured evidence bundle

## Retrieval Units

- static patch
- bi-temporal pair
- localized change region
- trajectory window

Each retrieval mode should index the correct unit rather than forcing one universal index.

For the paper direction, the preferred scientific unit is the bi-temporal pair plus its semantic transition summary.

## Evidence Flow

Preferred research flow:

`segmentation evidence -> retrieval evidence -> EvidenceBundle -> optional explainer`

The explainer must consume structured evidence only.

## Retrieval Head Position

The retrieval branch should share the same bi-temporal evidence model as the transition segmentation path.

```mermaid
flowchart LR
    PAIR["T1 / T2 image pair"] --> ENC["EO encoder + change fusion"]
    ENC --> Z["Global change embedding"]
    QUERY["Text query"] --> TXT["Text encoder"]
    TXT --> E["Text embedding"]
    Z --> PROJ["Shared embedding space"]
    E --> PROJ
    PROJ --> T2C["Text-to-change retrieval"]
    PROJ --> P2P["Pair-to-pair retrieval"]
```

## Shared Head Rule

One shared retrieval head is preferred for both:

- text-to-change retrieval
- pair-to-pair retrieval

That means the system should not add a separate pair-to-pair head unless experiments show a clear measurable gain.

## Novelty Guardrail

`retired visual baseline + pair fusion + contrastive retrieval` is a strong engineering baseline, but not a sufficient paper claim by itself.

The stronger claim is:

`transition-aware, direction-aware retrieval`

That means retrieval quality should improve when examples share:

- similar semantic transition histograms
- the same dominant transition direction
- localized change structure that matches the target transition
