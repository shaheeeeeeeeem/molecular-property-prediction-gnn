import os
import json
import pandas as pd

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.nn import GINConv, GATConv, global_mean_pool
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


def smiles_to_tensors(smiles):
    graph = smiles2graph(smiles)
    x = torch.tensor(graph["node_feat"], dtype=torch.long)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    return x, edge_index, batch


st.title("HIV Inhibition Prediction — GIN vs GAT")
st.caption("Predicts the probability that a molecule inhibits HIV replication, per the ogbg-molhiv benchmark.")

EXAMPLES = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
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
        col1.metric("GIN — HIV inhibition probability", f"{gin_prob:.4f}")
        col2.metric("GAT — HIV inhibition probability", f"{gat_prob:.4f}")
