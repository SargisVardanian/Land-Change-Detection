# Wrong-Project Changes Log

This file records the changes and task requests that were identified as belonging to another project and then reverted from the Land Change Detection app.

## User task requests that belonged to the other project

1. Build a text-first graph research workflow for political/process investigation.
2. Accept a text task in the web app and show graph expansion from web research rounds.
3. Show a lower process panel with:
   - tool usage;
   - opened sites;
   - temporary subgraph formation;
   - final merge into the main graph.
4. Run an experiment about chiefs of staff to Prime Minister Nikol Pashinyan in 2018-2026.
5. Save prompts, run artifacts, and performance notes for that political graph workflow.

## Changes that were made for the other project

1. Added a `graph_research` package under `src/land_change_detection/graph_research/`.
2. Added a text graph research mode to `app.py`.
3. Added prompt and performance documents:
   - `docs/graph_research_prompt.md`
   - `docs/graph_research_workflow.md`
   - `docs/graph_research_performance.md`
4. Added a replay fixture and experiment runner:
   - `data/graph_research/pashinyan_staff_2018_2026_fixture.json`
   - `scripts/run_graph_research_experiment.py`
5. Added experiment artifacts:
   - `artifacts/graph_research_runs/pashinyan_staff_2018_2026/*`
   - `artifacts/graph_research_runs/web_app_latest/*`
6. Added a graph-research test:
   - `tests/test_graph_research_pipeline.py`

## Reversion intent

The Land Change Detection project should return to the previous surface-change workflow and UI, without the political graph-research feature layer above.
