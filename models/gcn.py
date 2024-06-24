import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

EMBEDDING_DIM = 64

class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.initial_conv = GCNConv( # The graph convolutional operator from the “Semi-supervised Classification with Graph Convolutional Networks” paper
          in_channels=input_dim, # number of features per node of graph before transformation
          out_channels=EMBEDDING_DIM # number of features per node of graph after transformation
        )
        self.conv1 = GCNConv(EMBEDDING_DIM, EMBEDDING_DIM)
        self.conv2 = GCNConv(EMBEDDING_DIM, EMBEDDING_DIM)
        self.conv3 = GCNConv(EMBEDDING_DIM, EMBEDDING_DIM)
        self.out = Linear(
          in_features=EMBEDDING_DIM*2, # we stack the different global pooling aggregations below
          out_features=output_dim
        )

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)
          
        # Global Pooling (stack different aggregations over nodes of graph)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden



