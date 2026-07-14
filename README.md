# ogbg-molhiv: GIN vs GAT

GIN and GAT trained from scratch on OGB's `ogbg-molhiv` benchmark for HIV inhibition prediction, benchmarked against the public leaderboard.

Molecules are graphs, atoms are nodes, bonds are edges, so this is a natural fit for graph neural networks rather than the usual image/text/tabular setups. The point of this project is a direct, controlled comparison of two different message-passing schemes (GIN's sum-aggregation vs. GAT's attention) on the same real chemistry task, not just wrapping an existing library.

## Project structure

- `notebooks/01_data_exploration.ipynb` — loads the dataset via OGB's official loader and scaffold split, reports class balance, graph size stats, and node/edge feature semantics.
- `notebooks/02_gin_baseline.ipynb` — builds and tunes a GIN model, checkpoints on best validation ROC-AUC, saves `models/gin_best.pt`.
- `notebooks/03_gat_model.ipynb` — builds and tunes a GAT model under the same training setup, saves `models/gat_best.pt`.
- `notebooks/04_evaluation_comparison.ipynb` — loads both checkpoints, reports ROC-AUC and precision-recall curves, and compares against OGB's public leaderboard.
- `app/streamlit_app.py` — a small demo app: paste a SMILES string (or pick an example) and see both models' predicted HIV-inhibition probability side by side.

## Setup

```
pip install torch torch-geometric ogb rdkit
```

Each notebook has a commented-out Colab mount/install cell at the top — uncomment it if running on Colab, leave it commented for a local run. Every notebook auto-detects CUDA and falls back to CPU.

Run the notebooks in order (01 → 04). The dataset downloads automatically into `dataset/` on first run via OGB's loader.

To try the demo app after running notebooks 02 and 03:

```
streamlit run app/streamlit_app.py
```

## Results

| Model | Test ROC-AUC |
|---|---|
| GIN | 0.7598 |
| GAT | 0.7403 |

For context, OGB's official single-model baselines on this benchmark: GCN 0.7606, GIN 0.7558 (no virtual node). Top leaderboard entries reach ROC-AUC ~0.84-0.85 using ensembling, molecular fingerprints, and pretraining, techniques outside the scope of this project. Both models here omit edge features (bond type, stereo, conjugation) to keep the GIN/GAT comparison controlled — see notebook 03 for the reasoning.
