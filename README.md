# Unsupervised Anomaly Detection on the Elliptic Bitcoin Transaction Graph

A multi-method comparison of unsupervised anomaly detection techniques for identifying illicit Bitcoin transactions, using the [Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) (Weber *et al.*, KDD 2019).

Fraud labels are **never** used during training - only for evaluation. The project compares **tabular** baselines against a **graph-based** autoencoder and an **ensemble** that fuses all methods via rank averaging.

---

- ## Key Results (test set):

| Method | Test AUROC | Test AUPRC |
|--------|-----------|-----------|
| PCA Reconstruction Error | **0.7547** | 0.1739 |
| Graph Autoencoder (GCN) | 0.7337 | **0.2114** |
| Ensemble (average rank) | 0.7241 | 0.1209 |
| Isolation Forest | 0.6975 | 0.0889 |
| Local Outlier Factor | 0.3812 | 0.0371 |

- PCA leads on AUROC; the Graph Autoencoder leads on AUPRC (the more relevant metric for rare fraud at ~4.6% prevalence).
- All four base methods required **score polarity calibration** (automatic flipping when validation AUROC < 0.5).
- Every method shows a validation-to-test drop due to **temporal concept drift**.

---

- ## Project Structure:

```
elliptic_bitcoin_project/
├── notebooks/
│   ├── 01_data_exploration_and_preparation.ipynb   # EDA, time-based splits, graph construction
│   ├── 02_unsupervised_detection_and_evaluation.ipynb  # All 5 methods, metrics, plots
│   └── 03_report_and_interpretation.ipynb          # Written report, interpretation, conclusions
├── elliptic_bitcoin_dataset/                       # Raw CSVs (not tracked in git)
│   ├── elliptic_txs_features.csv
│   ├── elliptic_txs_classes.csv
│   └── elliptic_txs_edgelist.csv
├── artifacts/                                      # Saved bundles (not tracked in git)
│   ├── prep_bundle.pkl                             # Output of notebook 01
│   └── eval_bundle.pkl                             # Output of notebook 02
├── requirements-elliptic-uml.txt
├── .gitignore
└── README.md
```

---

- ## Methods:

### 1. Isolation Forest
Tree-based anomaly detector that isolates outliers through random feature splits. Anomaly score = negative of `score_samples` (higher = more suspicious).

### 2. PCA Reconstruction Error
PCA fitted with 30 components (~76% variance retained) on training data. The mean squared reconstruction error for each transaction serves as the anomaly score.

### 3. Local Outlier Factor (LOF)
Density-based method fitted with `novelty=True` and 20 neighbours on clean training data. Struggled in this 165-dimensional space due to the curse of dimensionality (test AUROC below random).

### 4. Graph Autoencoder (GCN + MLP)
A 2-layer GCN encoder aggregates neighbourhood information from the transaction graph, and an MLP decoder reconstructs each node's 165 features. Reconstruction error = anomaly score.
- **Dropout** (p=0.3) for regularization
- **Early stopping** (patience=20 epochs) to prevent overfitting
- Trained on the **train-induced subgraph**; inference uses the **full graph** (transductive)

### 5. Ensemble (average-rank fusion)
Scores from all four methods are converted to percentile ranks and averaged into a single combined score per transaction.

### Score polarity calibration
After scoring, validation AUROC is checked. If below 0.5 (scores pointing the wrong way), they are automatically flipped. All four base methods needed this correction.

---

- ## Pipeline Overview:

```
Raw CSVs
   │
   ▼
┌──────────────────────────┐
│  Notebook 01             │
│  - Load & explore data   │
│  - Time-based splits     │
│  - Build edge_index      │
│  - Save prep_bundle.pkl  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Notebook 02                     │
│  - Isolation Forest              │
│  - PCA Reconstruction Error      │
│  - LOF                           │
│  - Graph Autoencoder (GCN+MLP)   │
│  - Ensemble (rank fusion)        │
│  - Score polarity calibration    │
│  - Metrics + plots               │
│  - Save eval_bundle.pkl          │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Notebook 03                     │
│  - Load eval_bundle.pkl          │
│  - Results table & figures       │
│  - Interpretation                │
│  - Limitations & conclusions     │
└──────────────────────────────────┘
```

---

- ## Setup and Installation:

### Prerequisites
- Python 3.9+
- The Elliptic dataset CSVs placed in `elliptic_bitcoin_dataset/`

### Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements-elliptic-uml.txt
```

**PyTorch and PyTorch Geometric:** CPU builds work fine. For GPU support, follow [pytorch.org](https://pytorch.org/) and [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) before installing the requirements file.

### Run the notebooks (in order)

```bash
jupyter notebook
```

1. Open and run **`01_data_exploration_and_preparation.ipynb`** top to bottom (creates `artifacts/prep_bundle.pkl`)
2. Open and run **`02_unsupervised_detection_and_evaluation.ipynb`** top to bottom (creates `artifacts/eval_bundle.pkl`)
3. Open and run **`03_report_and_interpretation.ipynb`** top to bottom (loads results, displays report)

---

- ## Dataset:

The [Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) contains:
- **203,769 Bitcoin transactions** as nodes
- **234,355 directed edges** (payment flows)
- **165 features** per node (94 local + 71 aggregated neighbour features)
- **Labels:** 4,545 illicit (class 1), 42,019 licit (class 2), 157,205 unknown
- **49 time steps** used for temporal train/validation/test splitting

> Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.* KDD '19 Workshop on Anomaly Detection in Finance.

---

- ## Key Design Decisions:

| Decision | Rationale |
|----------|-----------|
| **Time-based splits** (not random) | Reflects real-world deployment where future transactions must be scored using models trained on past data |
| **Policy P1:** exclude known-illicit from `fit()` | Gives models cleaner "normal" statistics without using labels in the loss function |
| **Score polarity calibration** | Unsupervised scores have no guaranteed direction; flipping based on validation AUROC prevents silent inversions |
| **AUPRC alongside AUROC** | With ~4.6% illicit prevalence in test, AUPRC better reflects top-of-list precision |
| **Transductive inference for Graph AE** | Test nodes benefit from message-passing with training nodes via the full graph |

---

- ## Limitations:

- **Temporal concept drift** causes all methods to degrade from validation to test (up to -0.17 AUROC for Isolation Forest)
- **LOF fails** in 165 dimensions due to the curse of dimensionality
- **Single seed / fixed hyperparameters** - results are from one run without systematic tuning
- **Benchmark dataset only** - real-world deployment involves adversarial adaptation and regulatory constraints
- **Transductive Graph AE** has a structural advantage over tabular methods (test nodes receive messages from training nodes)

---

- ## Future Work:

- Systematic hyperparameter tuning using validation AUROC
- More expressive graph encoders (GAT, GraphSAGE)
- Graph-derived features (degree centrality, PageRank, clustering coefficient)
- Weighted or selective ensemble that down-weights poorly performing methods
- Cross-dataset evaluation on other temporal fraud benchmarks

---

- ## License:

This project was developed for academic coursework. The Elliptic dataset is subject to its own [license terms on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
