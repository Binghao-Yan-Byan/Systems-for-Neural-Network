import dgl
import torch
import torch.nn as nn
import torch.optim as optim 
from GCN_Byan import GCN_Byan
import scipy.sparse as sp
import math
import graphpy
import numpy as np

import argparse

def read_graph(dataset_name):
    if dataset_name == 'cora':
        data = dgl.data.CoraGraphDataset()
    elif dataset_name == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
    elif dataset_name == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
    elif dataset_name == 'reddit':
        data = dgl.data.RedditDataset()
    else:
        return None
    return data

def process_graph(g):
    col, row = g.edges(order='srcdst')
    numlist = torch.arange(col.size(0), dtype=torch.int32)
    adj_csr = sp.csr_matrix((numlist.numpy(), (row, col)), shape=(g.num_nodes(), g.num_nodes()))
    row_ptr = adj_csr.indptr.astype(np.int32)
    col_ind = adj_csr.indices.astype(np.int32)
    num_nodes = row_ptr.shape[0]-1
    degree = np.zeros(num_nodes).astype(np.int32)
    for i in range(num_nodes):
        if(row_ptr[i+1]-row_ptr[i]!=0):
            degree[i] = row_ptr[i+1] - row_ptr[i]
    g_csr = graphpy.init_graph(row_ptr, col_ind, degree)

    return g_csr


def train(hidden_feats, epoch_num, data, device):
    g = data[0] 
    g = dgl.add_self_loop(g)
    g_csr = process_graph(g)

    g = g.to(device)
    h = g.ndata['feat']
    labels = g.ndata['label']
    in_feats = h.shape[1]
    out_feats = data.num_classes
    train_mask = g.ndata['train_mask']
    
    model = GCN_Byan(in_features=in_feats, hidden_features=hidden_feats, out_features=out_feats, device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epoch_num):
        
        logits = model(g_csr, h)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epoch_num}, Loss:{loss.item()}")
        eval(model, g, g_csr)
    return model, g, g_csr

def eval(model, g, g_csr):
    h = g.ndata['feat']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']
    model.eval()
    with torch.no_grad():
        logits = model(g_csr, h)
        _, pred = logits[test_mask].max(dim=1)
        test_acc = (pred == labels[test_mask]).float().mean()
        print(f"Test Accuracy: {test_acc.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='select a dgl graph')
    parser.add_argument("--dataset", type=str, default="cora")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = read_graph(args.dataset)
    trained_model, g, g_csr = train(hidden_feats=32, epoch_num=50, data=data, device=device)
    eval(trained_model, g, g_csr)