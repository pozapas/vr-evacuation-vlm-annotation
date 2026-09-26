# Testing Vision-Language Model Annotations of Virtual Reality Evacuation Recordings Against the Game Engine Record

This repository holds the analysis code, the task descriptions, the analysis plan with its
amendments, every model response and the derived outputs behind the paper of the same title
(manuscript under review).

Eight vision-language models annotated 29 first-person recordings of a virtual reality fire
evacuation. The game engine logged the same sessions, so each annotation is checked against an
instrumented reference and not against a human coder. The engine gives three channels: event
markers, a 48 Hz gaze raycast that names the tagged object the participant looks at, and the
scene geometry.

## Contents

| Path | What it holds |
|---|---|
| `evactime/prompts/` | Task descriptions. `common_task_v2.md` is the original description and `common_task_v3_neutral.md` the neutral one used in the ablation. The condition files add the input format. |
| `evactime/outputs/*_raw.jsonl` | Every model call with its raw response, the parsed annotation, token use, latency and service route. |
| `evactime/outputs/` (other files) | Scored assertions, confusion matrices, permutation nulls, exit-claim labels, the ablation results and the aggregated human audit. |
| `evactime/outputs/filmstrip/` | The five frames shown in Figure 4. |
| `evactime/outputs/asset_hashes_*.csv` | SHA-256 of every frozen clip, muted clip and frame sent to the models. |
| `docs/NEUTRAL_PROMPT_PREREG.md` | Analysis plan for the neutral-prompt ablation, fixed before the first call, with Amendments 1 to 3. |
| `docs/CLASSIFIER_REVIEW_NEUTRAL.md` | Hand review of the exit-claim classifier on the neutral-prompt narratives. |
| `docs/litreview/grid_scores.csv`, `paper/supplement/positioning_evidence.csv` | Score, quotation and location for every cell of Table 1. |
| `paper/figures/q1_fig1_framework.drawio` | Source of Figure 2. |

## Scripts, in the order they run

| Step | Scripts | Needs the data deposit |
|---|---|---|
| Engine reference and scene geometry | `11_engine_reference.py`, `12_reference_final_engine.py`, `S1_scene_geometry.py`, `S2_reference_quality.py` | yes |
| Frozen model inputs (clips and frames) | `13_build_assets.py` | yes |
| Model calls, original description | `16_gemini_native.py` (native Gemini interface), `17_run_full_openrouter.py` (OpenRouter) | yes, plus API keys |
| Model calls, neutral-prompt ablation | `19_neutral_prompt.py`, `20_gemini_openrouter.py`, `21_v2_sameday.py`, `22_gemini_native_both.py` | yes, plus API keys |
| Attention scoring | `A1_attention_scoring.py`, `C2_resolution_limit.py`, `A2_scoring_null.py`, then `C2_confusion.py` (stratified null, bootstrap, per-target and per-model chance) | all but C2_confusion |
| Exit claims | `transit_classifier.py`, `C4_script_completion.py` (needs the deposit), `C6_neutral_prompt.py`; `S3_blind_validation_sheet.py` draws the blind validation sample and `S3b_score_blind_validation.py` scores the frozen classifier against the blind labels in `docs/classifier_blind_validation/` | C4 only |
| Human audit | `score_audit.py`, `C5_audit_analysis.py` | raw returns (not shared) |
| Tables | `Q1_tables.py`, `Q1_tab_models.py`, `Q1_tab_positioning.py`, `tabviz.py` | Q1_tables only |
| Figures | `Q1_figures.py` (Figure 1), `Q1_figures_v3.py` (Figure 3), `fig_script_v2.py` (Figure 4), `fig_prompt_ablation.py` (Figure 5), with `figstyle.py` and `palette.py` | no |
| Supplementary Material | `Q1_supplement.py` writes the supplement from the outputs | no |
| Number check | `Q1_numbers_audit.py` checks every number the paper states against the outputs | partly |

`vlm_records.py` loads and filters the model responses. `config.yaml` holds the frozen constants.

## Running it

```
pip install -r requirements.txt
py evactime/C6_neutral_prompt.py
py evactime/fig_prompt_ablation.py
```

Scripts marked "no" run from the files in this repository. The engine logs, gaze raycast and
screen recordings are in the data deposit (link in the paper). Place its folders next to
`evactime/` before running the scripts that need them. Model calls read `GEMINI_API_KEY` and
`OPENROUTER_API_KEY` from a `.env` file in the repository root, which is never committed.

## Not included

The screen recordings and engine logs are in the data deposit, not here. The full texts of the
papers in Table 1 are not redistributed, so the quotation check of that table cannot be rerun from
this repository. Raw audit returns are not shared, and coder identifiers are replaced by
pseudonyms (C01 to C11).

## License

Code is released under the MIT License. The model responses and derived outputs are released under
CC BY 4.0.
