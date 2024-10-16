
'''
GIN (Graph Isomorphsim Network) based odor classifier 
'''

import torch 
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.nn.models import GAT, GIN
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

class GraphClassifier(torch.nn.Module):
    """Module containing a node pooling layer and binary classifier head."""
    def __init__(self, in_channels, out_channels, pooling_type='mean'):
        super().__init__()
        self.mlp = torch.nn.Linear(in_channels, out_channels)
        if pooling_type == 'mean':
          self.pool = global_mean_pool
        elif pooling_type == 'max':
          self.pool = global_max_pool
        elif pooling_type == 'add':
          self.pool = global_add_pool

    def forward(self, x, batch):
        x = self.pool(x, batch)
        return torch.nn.functional.softmax(self.mlp(x), dim=1)
    def reset_parameters(self):
      self.mlp.reset_parameters()


class OdorGIN(torch.nn.Module):
    """Module containing a GNN with a binary classifier head."""
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels=138, dropout=0.2, pooling_type='mean'):
        super().__init__()
        self.gnn = GIN(
            in_channels, 
            hidden_channels, 
            num_layers,
            dropout=dropout,
        )
        self.classifier = GraphClassifier(hidden_channels, out_channels, pooling_type=pooling_type)
        self.gnn.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch=torch.tensor([0])):
        h = self.gnn(x, edge_index, edge_attr=edge_attr)
        return self.classifier(h, batch)


if __name__ == '__main__':
   pass