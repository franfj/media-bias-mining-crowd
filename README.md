# Can the Crowd Detect Bias? Interaction Signals as Proxies for Media Bias

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19164549.svg)](https://doi.org/10.5281/zenodo.19164549)

This repository contains the experiment pipeline and analysis code for our study on whether user interaction patterns on social news platforms carry complementary information for automated media bias detection.

## Key Findings

- **Biased articles produce a distinctive behavioral signature** across four independent feature families: karma distributions (higher entropy, +4.6%, p < 10⁻²⁴), comment sentiment (more anger, p < 10⁻⁵), temporal dynamics (10.7% faster reactions, 20.6% shorter threads), and network structure.
- **Interaction features alone achieve AUC 0.642** for bias classification using zero textual content, establishing baselines for future multimodal systems.
- **Two media ecosystems** emerge from community detection, organized by media format rather than ideology.
- **The signal is temporally robust**: consistent across 16 years of data (2006–2021).

## Dataset

**14,995 articles** × **38 features** × **13.2M comments** × **96K users** × **2,868 outlets** × **17 years** (2005–2021)

Available on Zenodo: [10.5281/zenodo.19164549](https://doi.org/10.5281/zenodo.19164549)

## Pipeline

The processing pipeline comprises 11 scripts, each producing self-contained intermediate outputs:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_ingest_dataset.py` | Convert raw JSON archive to Parquet (186K articles, 13.2M comments) |
| 2 | `02_filter_subsample.py` | Filter and stratified subsample (→ 14,995 articles) |
| 3 | `03_bias_labeling.py` | Automatic bias labeling with DistilBERT (MBBMD) |
| 4 | `04_interaction_features.py` | Extract 38 interaction features per article |
| 5 | `05_analysis.py` | Statistical analysis, correlations, predictive modeling |
| 6 | `06_karma_divergence.py` | Advanced karma distribution analysis (entropy, Gini, KL) |
| 7 | `07_user_media_graph.py` | Bipartite graph analysis, Louvain communities, user polarization |
| 8 | `08_comment_sentiment.py` | Comment sentiment and emotion analysis (robertuito) |
| 9 | `09_temporal_sentiment.py` | Intra-article dynamics and temporal trends |
| 10 | `10_train_bias_model.py` | Cross-lingual bias model training (DistilBERT + BEADS + MBBMD) |
| 11 | `11_relabel_with_new_model.py` | Re-label articles with new model + comparison |

Run the full pipeline:
```bash
cd experiments/scripts
bash run_pipeline.sh          # Steps 1-4
python 05_analysis.py         # Step 5
python 06_karma_divergence.py # Step 6
# ... etc.
```

## Repository Structure

```
├── experiments/
│   ├── scripts/          # 11 processing + analysis scripts
│   ├── data/             # Intermediate data (gitignored; available on Zenodo)
│   ├── results/          # Analysis outputs (CSV)
│   └── models/           # Trained models (gitignored)
├── PAPER_STATUS.md       # Project status and findings summary
└── README.md             # This file
```

## Results Summary

| Experiment | Key Result |
|-----------|------------|
| Karma entropy | +4.6% for biased articles (p < 10⁻²⁴) |
| Comment anger | +9.3% anger score (p < 4×10⁻⁵) |
| Reaction speed | 10.7% faster (p < 10⁻⁴) |
| Thread lifespan | 20.6% shorter (p < 10⁻⁴) |
| Karma drift | 9.1% steeper degradation (p < 10⁻¹²) |
| Classification (AUC) | 0.642 (Gradient Boosting, no text) |
| User communities | 2 outlet clusters (format-based, not ideology) |

## Citation

```bibtex
@article{rodrigo2026crowd,
  title={Can the Crowd Detect Bias? Interaction Signals as Proxies
         for Media Bias on Social News Platforms},
  author={Rodrigo-Gin{\'e}s, Francisco-Javier and
          Carrillo-de-Albornoz, Jorge and Plaza, Laura},
  journal={Procesamiento del Lenguaje Natural},
  year={2026}
}
```

## License

- **Code**: MIT
- **Dataset**: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
