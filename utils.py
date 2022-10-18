import torch
import random
import numpy as np
import os
from torch_geometric.utils import add_self_loops, sort_edge_index

# +
def build_fully_connected_edge_idx(num_nodes):
    fully_connected = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    _edge = np.where(fully_connected)
    edge_index = np.array([_edge[0], _edge[1]], dtype=np.int64)
    edge_index = sort_edge_index(torch.tensor(edge_index))[0]
    return edge_index


def build_batch_edge_index(edge_index, num_graphs, num_nodes):
    new_edge = edge_index
    for num in range(1, num_graphs):
        next_graph_edge = edge_index + num * num_nodes
        new_edge = torch.cat([new_edge, next_graph_edge], dim=-1)
    return new_edge


# -

def dice(pred, gt):
    XnY = torch.ones((len(gt))) * 32
    for i in range(len(gt)):
        if pred[i] == gt[i]:
            XnY[i] = pred[i]
    D = torch.zeros((32))
    for j in range(32):
        if (len(torch.where(pred == j)[0]) + len(torch.where(gt == j)[0])) == 0:
            D[j] = 0
        else:
            D[j] = ((2 * len(torch.where(XnY == j)[0])) / (
                        len(torch.where(pred == j)[0]) + len(torch.where(gt == j)[0])))

    dice = torch.sum(D) /32
    return dice

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
