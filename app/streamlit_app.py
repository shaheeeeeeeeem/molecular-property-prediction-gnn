import os
import json
import pandas as pd

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.nn import GINConv, GATConv, global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.utils.mol import smiles2graph

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GIN(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        h = self.atom_encoder(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = global_mean_pool(h, batch)
        return self.out(h).squeeze(-1)


class GAT(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, heads):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_dim = hidden_dim
        for i in range(num_layers):
            is_last = i == num_layers - 1
            concat = not is_last
            conv = GATConv(in_dim, hidden_dim, heads=heads, concat=concat, dropout=dropout)
            self.convs.append(conv)
            in_dim = hidden_dim * heads if concat else hidden_dim
            self.bns.append(nn.BatchNorm1d(in_dim))
        self.dropout = dropout
        self.out = nn.Linear(in_dim, 1)

    def forward(self, x, edge_index, batch):
        h = self.atom_encoder(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = global_mean_pool(h, batch)
        return self.out(h).squeeze(-1)


class GINVirtual(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.virtualnode_embedding = nn.Embedding(1, hidden_dim)
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.mlp_virtualnode_list = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.mlp_virtualnode_list.append(nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.BatchNorm1d(2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ))

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1, dtype=torch.long, device=x.device)
        )

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            h = self.convs[layer](h_list[layer], edge_index)
            h = self.bns[layer](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_list.append(h)

            if layer < self.num_layers - 1:
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                virtualnode_embedding = F.dropout(
                    self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                    p=self.dropout, training=self.training,
                )

        h_final = h_list[-1]
        h_pooled = global_mean_pool(h_final, batch)
        return self.out(h_pooled)


MOLPCBA_TASK_NAMES = [
    'PCBA-1030', 'PCBA-1379', 'PCBA-1452', 'PCBA-1454', 'PCBA-1457', 'PCBA-1458',
    'PCBA-1460', 'PCBA-1461', 'PCBA-1468', 'PCBA-1469', 'PCBA-1471', 'PCBA-1479',
    'PCBA-1631', 'PCBA-1634', 'PCBA-1688', 'PCBA-1721', 'PCBA-2100', 'PCBA-2101',
    'PCBA-2147', 'PCBA-2242', 'PCBA-2326', 'PCBA-2451', 'PCBA-2517', 'PCBA-2528',
    'PCBA-2546', 'PCBA-2549', 'PCBA-2551', 'PCBA-2662', 'PCBA-2675', 'PCBA-2676',
    'PCBA-411', 'PCBA-463254', 'PCBA-485281', 'PCBA-485290', 'PCBA-485294', 'PCBA-485297',
    'PCBA-485313', 'PCBA-485314', 'PCBA-485341', 'PCBA-485349', 'PCBA-485353', 'PCBA-485360',
    'PCBA-485364', 'PCBA-485367', 'PCBA-492947', 'PCBA-493208', 'PCBA-504327', 'PCBA-504332',
    'PCBA-504333', 'PCBA-504339', 'PCBA-504444', 'PCBA-504466', 'PCBA-504467', 'PCBA-504706',
    'PCBA-504842', 'PCBA-504845', 'PCBA-504847', 'PCBA-504891', 'PCBA-540276', 'PCBA-540317',
    'PCBA-588342', 'PCBA-588453', 'PCBA-588456', 'PCBA-588579', 'PCBA-588590', 'PCBA-588591',
    'PCBA-588795', 'PCBA-588855', 'PCBA-602179', 'PCBA-602233', 'PCBA-602310', 'PCBA-602313',
    'PCBA-602332', 'PCBA-624170', 'PCBA-624171', 'PCBA-624173', 'PCBA-624202', 'PCBA-624246',
    'PCBA-624287', 'PCBA-624288', 'PCBA-624291', 'PCBA-624296', 'PCBA-624297', 'PCBA-624417',
    'PCBA-651635', 'PCBA-651644', 'PCBA-651768', 'PCBA-651965', 'PCBA-652025', 'PCBA-652104',
    'PCBA-652105', 'PCBA-652106', 'PCBA-686970', 'PCBA-686978', 'PCBA-686979', 'PCBA-720504',
    'PCBA-720532', 'PCBA-720542', 'PCBA-720551', 'PCBA-720553', 'PCBA-720579', 'PCBA-720580',
    'PCBA-720707', 'PCBA-720708', 'PCBA-720709', 'PCBA-720711', 'PCBA-743255', 'PCBA-743266',
    'PCBA-875', 'PCBA-881', 'PCBA-883', 'PCBA-884', 'PCBA-885', 'PCBA-887',
    'PCBA-891', 'PCBA-899', 'PCBA-902', 'PCBA-903', 'PCBA-904', 'PCBA-912',
    'PCBA-914', 'PCBA-915', 'PCBA-924', 'PCBA-925', 'PCBA-926', 'PCBA-927',
    'PCBA-938', 'PCBA-995',
]


@st.cache_resource
def load_models():
    with open(f"{BASE_DIR}/outputs/results.json") as f:
        results = json.load(f)

    gin_model = GIN(**results["gin"]["config"]).to(device)
    gin_model.load_state_dict(torch.load(f"{BASE_DIR}/models/gin_best.pt", map_location=device))
    gin_model.eval()

    gat_model = GAT(**results["gat"]["config"]).to(device)
    gat_model.load_state_dict(torch.load(f"{BASE_DIR}/models/gat_best.pt", map_location=device))
    gat_model.eval()

    return gin_model, gat_model


@st.cache_resource
def load_molpcba_model():
    with open(f"{BASE_DIR}/outputs/results_molpcba_improved.json") as f:
        results = json.load(f)

    config = results["gin_virtual"]["config"]
    model = GINVirtual(out_dim=len(MOLPCBA_TASK_NAMES), **config).to(device)
    model.load_state_dict(torch.load(f"{BASE_DIR}/models/gin_molpcba_improved.pt", map_location=device))
    model.eval()
    return model


def smiles_to_tensors(smiles):
    graph = smiles2graph(smiles)
    x = torch.tensor(graph["node_feat"], dtype=torch.long, device=device)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long, device=device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    return x, edge_index, batch


st.title("HIV Inhibition Prediction — GIN vs GAT")
st.caption("Two graph neural networks predict whether a molecule inhibits HIV replication.")

with st.expander("What am I looking at?"):
    st.markdown(
        "Each molecule is converted into a graph (atoms as nodes, bonds as edges) and fed through "
        "two GNN architectures trained from scratch on OGB's `ogbg-molhiv` benchmark (41K molecules, "
        "~3.5% active). **GIN** scored 0.76 test ROC-AUC, **GAT** scored 0.74, evaluated against the "
        "official OGB leaderboard. Pick a molecule below (or paste your own SMILES) to see both models' "
        "predicted probability of activity. [Full writeup and code](https://github.com/shaheeeeeeeeem/molecular-property-prediction-gnn)."
    )

EXAMPLES = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Nicotine": "CN1CCCC1c1cccnc1",
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Phenol": "Oc1ccccc1",
    "Toluene": "Cc1ccccc1",
    "Naphthalene": "c1ccc2ccccc2c1",
    "Acetone": "CC(=O)C",
    "Urea": "NC(=O)N",
    "Methane": "C",
    "Water": "O",
}

choice = st.selectbox("Example molecule", ["Custom"] + list(EXAMPLES.keys()))
smiles = st.text_input("SMILES string", value=EXAMPLES.get(choice, ""))

models_available = os.path.exists(f"{BASE_DIR}/models/gin_best.pt") and os.path.exists(f"{BASE_DIR}/models/gat_best.pt")

if not models_available:
    st.warning("Trained model checkpoints not found. Run notebooks 02 and 03 first to produce `models/gin_best.pt` and `models/gat_best.pt`.")
elif smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string — could not parse this molecule.")
    else:
        st.image(Draw.MolToImage(mol, size=(350, 350)))

        gin_model, gat_model = load_models()
        x, edge_index, batch = smiles_to_tensors(smiles)

        with torch.no_grad():
            gin_prob = torch.sigmoid(gin_model(x, edge_index, batch)).item()
            gat_prob = torch.sigmoid(gat_model(x, edge_index, batch)).item()

        col1, col2 = st.columns(2)
        col1.metric("GIN", f"{gin_prob:.4f}")
        col2.metric("GAT", f"{gat_prob:.4f}")
        st.caption("Predicted probability the molecule inhibits HIV replication (0 = inactive, 1 = active). Real actives are rare (~3.5% of the training set), so most molecules should score low.")

        st.divider()
        st.subheader("Bioassay activity prediction (ogbg-molpcba)")

        molpcba_available = os.path.exists(f"{BASE_DIR}/models/gin_molpcba_improved.pt") and os.path.exists(f"{BASE_DIR}/outputs/results_molpcba_improved.json")

        if not molpcba_available:
            st.warning("molpcba model checkpoint not found. Run notebook 07 first to produce `models/gin_molpcba_improved.pt`.")
        else:
            molpcba_model = load_molpcba_model()
            with torch.no_grad():
                molpcba_probs = torch.sigmoid(molpcba_model(x, edge_index, batch)).squeeze(0).cpu().numpy()

            n_active = int((molpcba_probs > 0.5).sum())
            st.metric("Predicted active against", f"{n_active} of {len(MOLPCBA_TASK_NAMES)} assay targets")

            top_k = 10
            top_indices = molpcba_probs.argsort()[::-1][:top_k]
            table = pd.DataFrame({
                "Assay": [f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{MOLPCBA_TASK_NAMES[i].split('-')[1]}" for i in top_indices],
                "Probability": [float(molpcba_probs[i]) for i in top_indices],
            })
            st.dataframe(
                table,
                column_config={
                    "Assay": st.column_config.LinkColumn("PubChem BioAssay", display_text=r"bioassay/(\d+)"),
                    "Probability": st.column_config.NumberColumn(format="%.4f"),
                },
                hide_index=True,
            )
            st.caption(
                "This model (GIN + virtual node, notebook 07) predicts activity against 128 PubChem bioassay "
                "targets at once. Showing the top 10 by predicted probability, labeled by PubChem BioAssay ID "
                "(AID) rather than a plain-English description, since full assay descriptions aren't available "
                "locally. Click a row to look up what that assay actually measures."
            )
