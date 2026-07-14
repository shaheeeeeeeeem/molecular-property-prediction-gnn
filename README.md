# ogbg-molhiv: GIN vs GAT

GIN and GAT trained from scratch on OGB's `ogbg-molhiv` benchmark for HIV inhibition prediction, benchmarked against the public leaderboard.

Molecules are graphs, atoms are nodes, bonds are edges, so this is a natural fit for graph neural networks rather than the usual image/text/tabular setups. The point of this project is a direct, controlled comparison of two different message-passing schemes (GIN's sum-aggregation vs. GAT's attention) on the same real chemistry task, not just wrapping an existing library.

## Project structure

- `notebooks/01_data_exploration.ipynb` — loads the dataset via OGB's official loader and scaffold split, reports class balance, graph size stats, and node/edge feature semantics.
- `notebooks/02_gin_baseline.ipynb` — builds and tunes a GIN model, checkpoints on best validation ROC-AUC, saves `models/gin_best.pt`.
- `notebooks/03_gat_model.ipynb` — builds and tunes a GAT model under the same training setup, saves `models/gat_best.pt`.
- `notebooks/04_evaluation_comparison.ipynb` — loads both checkpoints, reports ROC-AUC and precision-recall curves, and compares against OGB's public leaderboard.
- `notebooks/06_molpcba_multitask.ipynb` — extends the same GIN/GAT architectures to OGB's `ogbg-molpcba` benchmark, a 128-task multi-label bioassay dataset, with a masked multi-task loss.
- `notebooks/07_molpcba_improved.ipynb` — adds a virtual node to the GIN architecture and trains for 100 epochs, closing most of the gap left by notebook 06's shorter run.
- `app/streamlit_app.py` — a small demo app: paste a SMILES string (or pick an example) and see both models' predicted HIV-inhibition probability, plus the top predicted bioassay targets from the `molpcba` model.

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

## molpcba extension

`ogbg-molpcba` extends the same GIN/GAT comparison to a harder task: 128 binary bioassay activity labels per molecule instead of one, with about 39% of labels missing per molecule. Training uses a masked loss that averages BCE per task and skips tasks with no valid labels in a given batch, so missing data doesn't distort the gradient. The official metric is mean Average Precision (AP) across all 128 tasks, since ROC-AUC isn't OGB's ranking metric here.

| Model | Test mean AP |
|---|---|
| GIN, no virtual node (notebook 06, 20 epochs) | 0.1492 |
| GAT, no virtual node (notebook 06, 20 epochs) | 0.1179 |
| GIN + virtual node (notebook 07, 100 epochs) | 0.2204 |
| OGB official — GIN, no virtual node | 0.2266 |
| OGB official — GIN + virtual node | 0.2703 |

Notebook 06's shorter run undertrained both models. Notebook 07 adds a virtual node to GIN and trains to 100 epochs, closing nearly all the gap to OGB's no-virtual-node baseline. The remaining gap to OGB's own virtual-node baseline comes down to specific things notebook 07 doesn't do: no edge-feature embedding (`BondEncoder`), no residual connections, and hyperparameters carried over from `molhiv` rather than tuned for this task. See notebook 07 for the full breakdown.
