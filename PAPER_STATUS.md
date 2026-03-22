# Paper Status — Perceived Media Bias in User Interactions

## Status: IN PROGRESS (Phase 2-4: Data Processing + Experiments)

| Field | Value |
|-------|-------|
| **Working title** | Perceived Media Bias in User Interactions: Mining the Crowd for Disinformation Signals |
| **Authors** | Francisco-Javier Rodrigo-Gines (UNED), Jorge Chamorro-Padial (U. Lleida), Jorge Carrillo-de-Albornoz (UNED), Laura Plaza (UNED) |
| **Affiliation** | UNED — NLP & IR Group; Universitat de Lleida |
| **Primary venue** | IJIMAI (International Journal of Interactive Multimedia and Artificial Intelligence) |
| **Paper type** | Full paper (data + experiments) |
| **Estimated length** | 12-15 pages (journal) |
| **Language** | English |
| **GitHub repo** | https://github.com/franfj/media-bias-mining-crowd |
| **Dataset source** | Meneame scraped dataset (186,318 JSON files, ~2008-2022) |

---

## Core Idea

Investigate whether user interaction patterns on Meneame (Spain's largest social news aggregator) correlate with media bias detected via NLP models. If behavioral signals (comments, votes, controversy) can proxy for perceived bias, this enables interaction-aware bias detection at scale without manual annotation.

---

## Research Questions

**RQ1**: Can user interaction patterns on social news platforms (votes, comments, karma) serve as reliable proxy signals for perceived media bias?

**RQ2**: To what extent do user-generated bias signals correlate with article-level bias labels produced by automated classifiers?

**RQ3**: Do different bias levels elicit different engagement patterns across news sources, political topics, or time periods?

**RQ4**: Can user interaction signals be used as weak supervision for bias detection, reducing annotation cost while maintaining detection quality?

---

## Contributions (planned)

1. A large-scale dataset combining Spanish news content + user interaction metadata from Meneame (~15K articles)
2. Automatic bias labeling via supervised model (franfj/fdtd_media_bias_E) and zero-shot LLM validation
3. Correlation and predictive analysis: engagement metrics as bias proxies
4. Comparative evaluation: interaction-only vs. content-only vs. combined feature models for bias detection
5. Temporal and outlet-level analysis of bias-interaction relationships

---

## Draft Status

| Section | Status |
|---------|--------|
| Abstract | Written (preliminary) |
| Introduction | Written (preliminary) |
| Related Work | Not started |
| Dataset | Not started |
| Methods | Not started |
| Results | Not started |
| Discussion | Not started |
| Conclusion | Not started |

---

## Phase Progress

- [x] Phase 0: Idea validation and project setup
- [x] Phase 1: Literature review (in user-interactions-bias project)
- [>] Phase 2: Dataset processing and subsampling
- [>] Phase 3: Automatic bias labeling (supervised + zero-shot)
- [>] Phase 4: Interaction feature extraction
- [ ] Phase 5: Analysis and experiments
- [ ] Phase 6: Paper writing (full draft)
- [ ] Phase 7: GitHub repo + Zenodo dataset upload
- [ ] Phase 8: Review and submission

---

## Data Pipeline

| Step | Script | Status |
|------|--------|--------|
| 1. Ingest JSON files to Parquet | `experiments/scripts/01_ingest_dataset.py` | In progress |
| 2. Filter and subsample | `experiments/scripts/02_filter_subsample.py` | In progress |
| 3. Bias labeling (supervised) | `experiments/scripts/03_bias_labeling.py` | In progress |
| 4. Interaction features | `experiments/scripts/04_interaction_features.py` | In progress |
| 5. Analysis | `experiments/scripts/05_analysis.py` | Not started |

---

## Next Actions

1. Complete dataset ingestion and subsampling pipeline
2. Run supervised bias labeling with franfj/fdtd_media_bias_E
3. Extract interaction features for all articles
4. Run correlation analysis
5. Write dataset and methods sections
