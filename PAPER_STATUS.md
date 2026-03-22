# Paper Status — Perceived Media Bias in User Interactions

## Status: IN PROGRESS (Phase 5: Analysis + Experiments)

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
- [x] Phase 2: Dataset processing and subsampling (186K → 168K filtered → 15K subsample)
- [x] Phase 3: Automatic bias labeling (franfj/fdtd_media_bias_E: 61.5% biased, 38.5% non-biased)
- [x] Phase 4: Interaction feature extraction (38 columns, 17 interaction features)
- [ ] Phase 5: Analysis and experiments
- [ ] Phase 6: Paper writing (full draft)
- [ ] Phase 7: GitHub repo + Zenodo dataset upload
- [ ] Phase 8: Review and submission

---

## Data Pipeline

| Step | Script | Status |
|------|--------|--------|
| 1. Ingest JSON files to Parquet | `experiments/scripts/01_ingest_dataset.py` | Done (186,317 articles, 13.2M comments) |
| 2. Filter and subsample | `experiments/scripts/02_filter_subsample.py` | Done (168,688 filtered → 14,995 subsample) |
| 3. Bias labeling (supervised) | `experiments/scripts/03_bias_labeling.py` | Done (9,225 biased / 5,770 non-biased) |
| 4. Interaction features | `experiments/scripts/04_interaction_features.py` | Done (38 columns incl. 17 interaction features) |
| Pipeline runner | `experiments/scripts/run_pipeline.sh` | Done (supports --from N, --only N) |
| 5. Analysis | `experiments/scripts/05_analysis.py` | Not started |

---

## Data Summary

| Dataset | Rows | Columns | Size |
|---------|------|---------|------|
| `articles_raw.parquet` | 186,317 | 15 | 89.4 MB |
| `comments_raw.parquet` | 13,252,609 | 7 | 2,114.6 MB |
| `articles_filtered.parquet` | 168,688 | 15 | 83.9 MB |
| `articles_subsample.parquet` | 14,995 | 15 | 7.8 MB |
| `articles_labeled.parquet` | 14,995 | 17 | 7.9 MB |
| `articles_with_features.parquet` | 14,995 | 38 | 8.7 MB |

### Bias Label Distribution (subsample)
- Biased (1): 9,225 (61.5%)
- Non-biased (0): 5,770 (38.5%)
- Mean bias probability: 0.616

---

## Next Actions

1. Run correlation analysis (bias labels vs interaction features)
2. Train interaction-only, content-only, and combined models
3. Perform temporal and outlet-level analysis
4. Write dataset and methods sections
5. Write results and discussion
