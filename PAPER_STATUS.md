# Paper Status — Perceived Media Bias in User Interactions

## Status: IN PROGRESS (Phase 5: Analysis + Experiments)

| Field | Value |
|-------|-------|
| **Working title** | Perceived Media Bias in User Interactions: Mining the Crowd for Disinformation Signals |
| **Authors** | Francisco-Javier Rodrigo-Gines (UNED), Jorge Chamorro-Padial (U. Lleida), Jorge Carrillo-de-Albornoz (UNED), Laura Plaza (UNED) |
| **Affiliation** | UNED — NLP & IR Group; Universitat de Lleida |
| **Paper type** | Full paper (data + experiments) |
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
- [x] Phase 5: Analysis and experiments (6 experiment scripts completed)
- [ ] Phase 6: Paper writing (full draft)
- [x] Phase 7: GitHub repo + Zenodo dataset upload (deposit 19164549, draft v0.1.0)
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
| 5. Statistical analysis | `experiments/scripts/05_analysis.py` | Done (correlations, prediction, outlet/topic/temporal) |
| 6. Karma divergence | `experiments/scripts/06_karma_divergence.py` | Done (entropy, gini, bimodality, KL per outlet) |
| 7. User-media graph | `experiments/scripts/07_user_media_graph.py` | Done (Louvain communities, Jaccard, user polarisation) |
| 8. Comment sentiment | `experiments/scripts/08_comment_sentiment.py` | Done (20K sample, pysentimiento/robertuito) |
| 9. Temporal dynamics | `experiments/scripts/09_temporal_sentiment.py` | Done (intra-article, yearly trends, sentiment) |
| 10. Bias model training | `experiments/scripts/10_train_bias_model.py` | Done (DistilBERT multi, F1=0.79 on BEADS val) |
| Zenodo upload | `experiments/scripts/upload_zenodo.py` | Done (deposit 19164549, draft v0.1.0) |

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

## Key Findings (Phase 5)

### Statistical Analysis (script 05)
- Gradient Boosting best predictor: AUC 0.642, F1 0.567 (baseline 0.615)
- `text_length` most predictive feature (Spearman ρ=0.20)
- Most biased topics: entrevista, curiosidades, ETA, iglesia
- Least biased: Microsoft, software libre, UE, Linux

### Karma Divergence (script 06)
- Biased articles have higher karma entropy (+4.6%, p<1e-24)
- Higher IQR (+10.5%) and extreme karma fraction (+2.0%)
- Outlets with highest KL divergence: danielmarin, escolar.net, labrujulaverde

### User-Media Graph (script 07)
- 2 media outlet communities (Louvain): traditional vs digital/alternative
- Most similar pair: elConfidencial ↔ elDiario (Jaccard 0.40)
- 45% of users have high bias exposure; mixed users read 3× more outlets

### Comment Sentiment (script 08)
- More negativity in biased articles: 59.7% NEG vs 56.2% (p<1e-5)
- More anger: 0.271 vs 0.248 (p<4e-5)
- Pattern consistent across 2006-2021

### Temporal Dynamics (script 09)
- Biased articles trigger 10.7% faster initial reactions
- Stronger karma drift (-8.4 vs -7.7, p<1e-12)
- Late comments 4.5% shorter; article life 20.6% shorter

### Bias Model (script 10)
- DistilBERT multilingual on BEADS+MBBMD: F1=0.79 on validation
- Cross-lingual transfer ES: 0% biased recall on MBBMD test (only 2 samples)
- Need more Spanish training data or translation augmentation

## Zenodo Dataset

- **Deposit ID**: 19164549
- **Status**: Draft (v0.1.0)
- **Files**: articles_with_features, articles_labeled, karma_features, comments_with_sentiment, user_profiles, user_outlet_interactions

## Next Actions

1. Improve bias model: add MBIC, try data augmentation / translation for Spanish
2. Re-label Meneame articles with improved model
3. Update Zenodo deposit with new labels
4. Write paper sections: Dataset, Methods, Results, Discussion
5. Publish Zenodo deposit with DOI
