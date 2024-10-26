import dgl
import torch
import torch.nn as nn
import torch.optim as optim 
from GCN import GCN

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

def train(hidden_feats, epoch_num, data, device = torch.device('cpu')):
    g = data[0]
    g = dgl.add_self_loop(g)
    g = g.to(device)

    h = g.ndata['feat']
    labels = g.ndata['label']
    in_feats = h.shape[1]
    out_feats = data.num_classes
    train_mask = g.ndata['train_mask']
    
    model = GCN(in_feats=in_feats, hidden_feats=hidden_feats, out_feats=out_feats)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epoch_num):
        model.train()
        logits = model(g, h)

        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epoch_num}, Loss:{loss.item()}")
        eval(model, g)

    return model, g

def eval(model, g):
    h = g.ndata['feat']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']
    model.eval()

    with torch.no_grad():
        logits = model(g, h)
        _, pred = logits[test_mask].max(dim=1)
        test_acc = (pred == labels[test_mask]).float().mean()
        print(f"Test Accuracy: {test_acc.item():.4f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='select a dgl graph')
    parser.add_argument("--dataset", type=str, default="cora")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = read_graph(args.dataset)
    trained_model, g = train(hidden_feats=32, epoch_num=50, data=data)
    eval(trained_model, g)