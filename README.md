# Perceived Media Bias in User Interactions: Mining the Crowd for Disinformation Signals

This repository contains the experiment pipeline and paper draft for our study on the relationship between media bias and user engagement patterns on [Meneame](https://www.meneame.net/), Spain's largest social news aggregator.

## Overview

We investigate whether user interaction patterns (votes, comments, karma, controversy) can serve as proxy signals for perceived media bias. The pipeline combines large-scale content analysis with engagement metadata, using a supervised DistilBERT classifier ([franfj/fdtd_media_bias_E](https://huggingface.co/franfj/fdtd_media_bias_E)) trained on the MBBMD dataset to label articles, and then correlates bias predictions with behavioral signals.

### Research Questions

- **RQ1**: Can user interaction patterns on social news platforms serve as reliable proxy signals for perceived media bias?
- **RQ2**: To what extent do user-generated bias signals correlate with article-level bias labels produced by automated classifiers?
- **RQ3**: Do different bias levels elicit different engagement patterns across news sources, political topics, or time periods?
- **RQ4**: Can user interaction signals be used as weak supervision for bias detection?

## Repository Structure

```
.
├── experiments/
│   ├── scripts/             # Processing and analysis pipeline
│   │   ├── 01_ingest_dataset.py       # Convert raw JSON to Parquet
│   │   ├── 02_filter_subsample.py     # Filter and stratified subsample
│   │   ├── 03_bias_labeling.py        # Automatic bias labeling (DistilBERT)
│   │   ├── 04_interaction_features.py # Extract engagement features
│   │   └── 05_analysis.py            # Statistical analysis and modeling
│   ├── data/                # Intermediate data (gitignored, see below)
│   └── results/             # Analysis output (CSV summaries)
├── drafts/
│   └── v1/                  # Paper draft (IJIMAI format)
├── literature/              # Literature review notes
├── notes/                   # Research notes
└── PAPER_STATUS.md          # Current project status
```

## Running the Pipeline

### Prerequisites

```bash
pip install pandas pyarrow numpy torch transformers scipy scikit-learn
```

### Step-by-step

1. **Ingest the raw Meneame dataset** (JSON files to Parquet):
   ```bash
   python experiments/scripts/01_ingest_dataset.py \
       --input-dir /path/to/meneame/dataset/_test/ \
       --output-dir experiments/data/
   ```

2. **Filter and subsample** (~15K articles from news outlets, stratified by year):
   ```bash
   python experiments/scripts/02_filter_subsample.py --data-dir experiments/data/
   ```

3. **Automatic bias labeling** using the [franfj/fdtd_media_bias_E](https://huggingface.co/franfj/fdtd_media_bias_E) model:
   ```bash
   python experiments/scripts/03_bias_labeling.py --data-dir experiments/data/ --device mps
   ```
   Supports `cpu`, `cuda`, or `mps` (Apple Silicon).

4. **Extract interaction features** (comment karma, polarization, engagement depth, etc.):
   ```bash
   python experiments/scripts/04_interaction_features.py --data-dir experiments/data/
   ```

5. **Run analysis** (statistical tests, correlations, outlet/topic/temporal analysis, predictive modeling):
   ```bash
   python experiments/scripts/05_analysis.py --data-dir experiments/data/
   ```
   Results are saved as CSV files in `experiments/results/`.

## Data Requirements

The raw dataset is **not included** in this repository due to its size (~186K JSON files, ~2 GB compressed).

The Meneame dataset consists of scraped news submissions from meneame.net (approximately 2008-2022). Each JSON file contains:
- Article metadata (title, text, URL, media outlet, score, tags, timestamp)
- User comments (text, author, karma, timestamp)

To reproduce the experiments, place the raw JSON files in a directory and point `01_ingest_dataset.py` to it.

## Target Venue

**IJIMAI** (International Journal of Interactive Multimedia and Artificial Intelligence)

## Authors

- **Francisco-Javier Rodrigo-Gines** - UNED, NLP & IR Group (frodrigo@invi.uned.es)
- **Jorge Carrillo-de-Albornoz** - UNED, NLP & IR Group
- **Laura Plaza** - UNED, NLP & IR Group

## License

This code is provided for research purposes. Please cite our paper if you use this pipeline.
