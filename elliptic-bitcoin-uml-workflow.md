# Elliptic Bitcoin Dataset — Unsupervised ML Execution Workflow

**Project root:** `elliptic_bitcoin_project/` (open your terminal and your IDE **here**).  
**Data directory:** `elliptic_bitcoin_project/elliptic_bitcoin_dataset/` — raw CSVs only.  
**Goal:** End-to-end **authentic UML** pipeline—**no fraud labels in training**—with **held-out evaluation** on licit vs illicit.  
**Last updated:** April 2026

---

## 1. Principles (authentic UML)

| Rule | What to do |
|------|------------|
| **Training** | Fit anomaly / representation models using **features + graph** only. **Do not** pass `class` (illicit/licit) into `fit()` or into the loss as a target. |
| **Unknown rows** | `class == unknown` may be used in training (they have no fraud label) or excluded—**state your choice** and keep it consistent. |
| **Evaluation** | Use **licit (2)** vs **illicit (1)** labels **only** on **validation** (for threshold/calibration) and **test** (for reported AUROC/AUPRC). **Exclude `unknown`** from classification metrics. |
| **Leakage** | Do not pick the final threshold on the **test** set. Prefer **time-based** splits aligned with column **time step** (features col 1). |

---

## 2. Requirements

### 2.1 Software

- **Python** 3.10+ (3.11 recommended)
- **Git** (optional, for versioning your notebook/repo)

### 2.2 Python packages (install in a virtual environment)

A starter file is in this folder: **`requirements-elliptic-uml.txt`**. Copy/rename to `requirements.txt` or install manually:

```text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.11
matplotlib>=3.7
```

**Optional (graph + deep UML):**

```text
torch>=2.0
torch-geometric>=2.4
pygod>=1.0
```

**Notes**

- Start with **pandas + sklearn** only if you want a **tabular baseline** first (Isolation Forest / PCA reconstruction on node features), then add **PyTorch Geometric** for a **graph** model.
- **PyGOD** needs PyTorch + usually PyG; install versions that match your CUDA/CPU [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 2.3 Hardware expectations

| Setup | Expectation |
|-------|-------------|
| **CPU only** | Baseline sklearn + **subsampled** graph or **full** graph with sparse ops—may be slow for large GNN epochs. |
| **GPU** | Speeds up **graph autoencoder / GNN** training noticeably. |

### 2.4 Data files (verify before coding)

All under **`<project_root>/elliptic_bitcoin_dataset/`**:

| File | Expected |
|------|----------|
| `elliptic_txs_features.csv` | **No header**; col `0` = `txId`, col `1` = time step, cols `2..166` = **165** numeric features (**167** columns total). |
| `elliptic_txs_classes.csv` | Header `txId,class`; values `unknown`, `1` (illicit), `2` (licit). |
| `elliptic_txs_edgelist.csv` | Header `txId1,txId2`; directed edges. |

**Sanity counts (reference):** ~203,769 nodes; ~234,355 edges; class counts ≈ unknown 157,205 / licit 42,019 / illicit 4,545.

---

## 3. Master to-do checklist

Use this as a **project board**; check off as you go.

- [ ] **T1** — Create venv, install `requirements-elliptic-uml.txt`, run `python -c "import pandas, sklearn"` (and torch/pyg if used).
- [ ] **T2** — Run **data verification script** (row counts, column count 167, class distribution, edge count).
- [ ] **T3** — Build **single `txId` → integer index** mapping; **consistent** node order for `X` and edges.
- [ ] **T4** — Implement **time-based train / val / test** (document cutoffs in time steps).
- [ ] **T5** — **Baseline UML model** (e.g. Isolation Forest or PCA + reconstruction error) **fit on train nodes only** (per protocol below).
- [ ] **T6** — **Graph UML model** (e.g. graph AE or PyGOD detector) **fit on train**; output anomaly score per node.
- [ ] **T7** — **Threshold** chosen on **val** only (or unsupervised percentile rule—document).
- [ ] **T8** — **Test metrics:** AUROC, AUPRC on **licit vs illicit** only; optional precision@k.
- [ ] **T9** — **Qualitative:** top-k suspicious tx + ego-net or feature summary.
- [ ] **T10** — **Report:** protocol diagram, limitations, ethics, **citation** (Weber et al., KDD 2019).

---

## 4. Execution phases (step by step)

### Phase A — Environment & repository layout

1. Open terminal in **`elliptic_bitcoin_project`** (your project root), e.g.  
   `f:\msc ds\semester 2\UML\uml project\elliptic_bitcoin_project`
2. Create venv: `python -m venv .venv` → activate (Windows: `.venv\Scripts\Activate.ps1`).
3. `pip install -r requirements-elliptic-uml.txt` (add `jupyter` for classic Notebook UI; uncomment optional PyTorch/PyG lines if needed).
4. Run **`notebooks/01_data_exploration_and_preparation.ipynb`** — verification + merge + splits + `artifacts/prep_bundle.pkl`.
5. Run **`notebooks/02_unsupervised_detection_and_evaluation.ipynb`** — models and metrics.
6. Suggested layout:
   - `elliptic_bitcoin_dataset/` — raw CSVs (read-only)
   - `notebooks/` — **all** pipeline code (no required `.py` scripts)
   - `artifacts/` — `prep_bundle.pkl`, plots (gitignore if large)

**Exit criteria:** Imports work; you can `pd.read_csv` the three files without errors.

---

### Phase B — Load and join tables

Use path prefix **`elliptic_bitcoin_dataset/`** from project root (or `DATA_DIR` in code).

1. **Classes:** `pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_classes.csv")` → columns `txId`, `class`.
2. **Features:** `pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_features.csv", header=None)` → name columns `txId`, `time_step`, `f_0`…`f_164` (or `range(167)`).
3. **Inner merge** `features` with `classes` on `txId` → one row per labeled node with features + `class`.
4. **Edges:** `pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")` → `txId1`, `txId2`.

**Expectations**

- Row count after merge ≈ **203,769** (if all ids align).
- No duplicate `txId` in features.

**Optional QC**

- Histogram of `time_step` (expect 1…49 over full data).
- Count edges whose endpoints are missing from `txId` set; **drop or keep**—document (small count is normal).

---

### Phase C — Graph index and tensors

1. Collect **all** `txId` values that appear in **features** (and optionally require presence in edge list—**define policy**).
2. Build `id2idx: dict` and `idx2id: list` of length **N**.
3. Remap edgelist:
   - `u = id2idx[txId1]`, `v = id2idx[txId2]`
   - Build **`edge_index`** shape `(2, E)` for PyG: `torch.tensor([u_list, v_list], dtype=torch.long)`.
4. Build **feature matrix** `X` shape `(N, F)` with **F = 165** (exclude `txId` and `time_step` from `X` unless you justify including time as a feature—common to **exclude time from X** for some splits and use time only for splitting).

**Expectations**

- `X[i]` corresponds to node index `i` for the same `txId`.
- `edge_index` uses **0-based** indices.

**Pitfall:** Using test nodes’ edges during training in a **transductive** GNN on the full graph can **leak** test structure. For strict UML reporting, either:
- **Inductive:** build subgraph **train nodes only** for training, or  
- **Transductive:** state clearly that the full graph was visible and focus evaluation on **held-out node labels** with a **time split** so test nodes are **future** time steps.

---

### Phase D — Train / validation / test split (time-based, recommended)

1. Sort or group by **`time_step`**.
2. Choose cutoffs, e.g.:
   - **Train:** time steps **1–T_train**
   - **Val:** **T_train+1–T_val**
   - **Test:** **T_val+1–49**  
   (Adjust fractions, e.g. ~70% / 15% / 15% of **time steps**, not necessarily of rows.)
3. For **each split**, mask nodes by their `time_step`.

**Training set policy (pick one, document)**

- **Policy P1 (strict):** Train UML models using **only train-time** nodes with `class in {unknown, 2}` — **exclude illicit (1)** from training so the model never sees fraud patterns in `fit`.
- **Policy P2 (looser):** Train on **all train-time** nodes including illicit—still **no gradient on labels**, but the representation may see illicit structure; disclose this.

**Evaluation subsets**

- **Val / test metrics:** restrict to nodes where `class` is **`1` or `2`** (drop `unknown`).

**Expectations**

- Test set contains **both** licit and illicit (check counts); if illicit is **zero** in test, **change cutoffs**.

---

### Phase E — Baseline unsupervised model (mandatory)

**Purpose:** Something simple and **label-free** to compare against graph methods.

**Example A — Isolation Forest**

- Input: `X_train` (standardize with `StandardScaler` **fit on train only**).
- `fit(X_train)` — no `y`.
- `score_samples` or `decision_function` → convert to **higher = more anomalous** if needed.

**Example B — PCA reconstruction error**

- `PCA` fit on train standardized `X_train`; reconstruction error on all nodes for scoring.

**Expectations**

- One **numpy/torch vector** `score_baseline[i]` per node index (or per train-fit scope—then broadcast consistently for eval).

---

### Phase F — Graph unsupervised model

**Purpose:** Use **edge_index** + **X** without labels.

**Options (choose one for scope)**

1. **PyGOD** detector on `torch_geometric.data.Data(x=..., edge_index=...)`, train on **train mask** only if API allows.
2. **Graph autoencoder** (encode → decode **X** or adjacency reconstruction); anomaly score = **reconstruction error**.
3. **Simpler graph-aware baseline:** append **1-hop aggregated neighbor mean** of features to `X`, then Isolation Forest (still UML; document as “hand-crafted graph features”).

**Expectations**

- Output `score_graph[i]` aligned with node index.
- Training **does not** use `class`.

---

### Phase G — Calibration (validation only)

1. On **val** nodes with `class in {1,2}`, choose threshold or rank cutoff:
   - **Unsupervised:** top **p%** highest scores as “flagged” (no val labels)—weak but label-pure.
   - **Weak supervision for ops:** maximize F1 on **val** using val labels **only for threshold**—state explicitly (“threshold tuned on validation fraud labels”).
2. **Do not** tune on **test**.

**Expectations**

- One clear **rule** documented in the report.

---

### Phase H — Final evaluation (test only)

1. Restrict to **test** nodes with `class` in `{1,2}`.
2. Compute **AUROC**, **AUPRC** comparing scores to binary **illicit=1 vs licit=2**.
3. Optional: **precision@k** for top-k flagged transactions.

**Expectations**

- Report **class imbalance** (illicit rare); AUPRC often more informative than accuracy.

---

### Phase I — Interpretation & report

1. **Top-k** highest scores on test (or global): list `txId`, true class (for analysis), time step.
2. **Ego-network:** subgraph of 1-hop neighbors for 1–2 examples (visual or table).
3. **Failure analysis:** high-score licit, low-score illicit.
4. **Ethics:** research benchmark only; no deployment / legal claims; cite dataset and paper.

**Citation (minimum)**

```bibtex
@inproceedings{weber2019anti,
  title={Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics},
  author={Weber, Mark and others},
  booktitle={KDD},
  year={2019}
}
```

Data source: **Elliptic** via **Kaggle** `ellipticco/elliptic-data-set` (or your actual source).

---

## 5. Deliverables (what “done” looks like)

| Deliverable | Description |
|-------------|-------------|
| **Code** | Runnable script or notebook: load → split → train → scores → eval. |
| **Protocol doc** | ½–1 page: P1 vs P2, time cutoffs, what’s excluded from metrics. |
| **Results table** | Baseline vs graph: **Val** (threshold method) + **Test** AUROC/AUPRC. |
| **Figures** | Score distribution; optional PR curve; ego-net sketch. |
| **Limitations** | Leakage choices, unknown labels, class imbalance, heterophily. |

---

## 6. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| **No illicit in test slice** | Plot illicit count per time step before fixing splits. |
| **Huge RAM** | Use `scipy.sparse` / PyG sparse; or subgraph sample with documented seed. |
| **Transductive leakage** | Prefer **time split** + clear wording; or train on **train-induced subgraph**. |
| **Editor hangs on features CSV** | Do not open 660MB+ CSV in IDE; use pandas/chunks. |

---

## 7. Optional extensions (if time)

- Compare **Policy P1 vs P2** test AUROC (short ablation).
- **Ensemble:** average rank of baseline + graph scores.
- **Stability:** rerun with different seeds / subsample.

---

## 8. Quick reference — column cheat sheet

| Object | Source |
|--------|--------|
| Node id | `features[:, 0]` or column `txId` after naming |
| Time step | `features[:, 1]` |
| Features for ML | columns **2–166** (165 dims) |
| Licit | `class == 2` |
| Illicit | `class == 1` |
| Unknown | `class == "unknown"` (string) — **exclude from AUROC** |
| Directed edge | `txId1 → txId2` |

---

*This workflow assumes project root = `elliptic_bitcoin_project/` and data in `elliptic_bitcoin_dataset/`. Adjust time cutoffs and policies to match your course rubric.*
