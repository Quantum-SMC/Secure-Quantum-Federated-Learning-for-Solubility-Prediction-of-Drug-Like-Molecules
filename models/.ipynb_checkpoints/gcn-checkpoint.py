#!/usr/bin/env python
# coding: utf-8

# ## Credit 
# 
# https://medium.com/@tejpal.abhyuday/application-of-gnn-for-calculating-the-solubility-of-molecule-graph-level-prediction-8bac5fabf600

# ## Predicting molecule solubility using graph neural networks
# 
# The goal of using graph-based representations in the `torch_geometric` library is to enable the application of graph neural networks (GNNs) and other graph-based machine learning techniques for tasks such as predicting solubility or other chemical properties. GNNs can operate directly on these graph structures, taking into account the connectivity of atoms and the associated node and edge features to make predictions.
# 
# [Source: ChatGPT]

# ## The ESOL dataset
# 
# The ESOL dataset, which stands for "Extended Solubility", is a dataset commonly used in the field of cheminformatics and machine learning for predicting the solubility of chemical compounds. The dataset contains information about the solubility of various organic molecules in water.
# 
# 1. **Graph Representation**: Each chemical compound is represented as a graph, where atoms are nodes, and chemical bonds are edges. This graph representation captures the connectivity and structure of the molecule.
# 
# 2. **Node Features**: The node feature vectors, in this case, typically represent the properties of individual atoms within the molecule. These features can include:
# 
#    - **Atom Type**: Each atom is assigned a specific atom type based on its element (e.g., carbon, hydrogen, oxygen, etc.). This is often one-hot encoded or represented as a categorical feature.
# 
#    - **Atomic Charges**: The partial charges on each atom, which describe the distribution of electric charge within the molecule.
# 
#    - **Hybridization**: Information about the hybridization state of each atom (e.g., sp3, sp2, sp).
# 
#    - **Atomic Mass**: The mass of each atom.
# 
#    - **Formal Charge**: The formal charge on each atom.
# 
#    - **Other Atom-specific Properties**: Depending on the specific implementation, additional atom-specific properties may also be included as node features.
# 
# 3. **Edge Features**: In addition to node features, edge features can be included in the graph representation. These features typically describe the type of chemical bond between connected atoms (e.g., single, double, or triple bonds) and may also include bond distances or bond angles.
# 
# 4. **Graph Structure**: The graph structure itself is represented by adjacency matrices or edge lists, which define how atoms are connected by chemical bonds.
# 
# [Source: ChatGPT]

# In[1]:


from torch_geometric.datasets import MoleculeNet
 
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MoleculeNet.html#torch_geometric.datasets.MoleculeNet
DATA = MoleculeNet(root=".", name="ESOL")
DATA


# ## Investigate the dataset
# 
# 

# In[3]:


print("Dataset type: ", type(DATA))
print("Number of features per graph node: ", DATA.num_features)
print("Number of distinct target values (solubilities): ", DATA.num_classes)
print("Number of graphs: ", len(DATA))
print("Example graph: ", DATA[0])
print("Number of nodes in example graph: ", DATA[0].num_nodes)
print("Number of edges in example graph: ", DATA[0].num_edges)


# In[5]:


# nodes of example graph 
DATA[0].x # shape: [num_nodes, num_node_features]


# In[7]:


# the target value of the example graph is its solubility
DATA[0].y


# In[9]:


# the edges of the example graph are in sparse Coordinate Format (COO)
# (also called adjacency list: https://distill.pub/2021/gnn-intro/
DATA[0].edge_index.t() # shape [num_edges, 2]


# In[11]:


# edge attributes of example graph
DATA[0].edge_attr # shape [num_edges, num_edge_features]


# ## Visualize one of the molecules in the dataset 

# In[14]:


from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole

Chem.MolFromSmiles(DATA[0]["smiles"])


# ## Implement a Graph Convolutional Neural Network

# In[17]:


import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

EMBEDDING_DIM = 64

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        self.initial_conv = GCNConv( # The graph convolutional operator from the “Semi-supervised Classification with Graph Convolutional Networks” paper
          in_channels=DATA.num_features, # number of features per node of graph before transformation
          out_channels=EMBEDDING_DIM # number of features per node of graph after transformation
        )
        self.conv1 = GCNConv(EMBEDDING_DIM, EMBEDDING_DIM)
        self.conv2 = GCNConv(EMBEDDING_DIM, EMBEDDING_DIM)
        self.conv3 = GCNConv(EMBEDDING_DIM, EMBEDDING_DIM)
        self.out = Linear(
          in_features=EMBEDDING_DIM*2, # we stack the different global pooling aggregations below
          out_features=1
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

MODEL = GCN()
print(MODEL)
print("Number of parameters: ", sum(p.numel() for p in MODEL.parameters()))


# ## Train the Graph Convolutional Network 

# In[20]:


from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

NUM_GRAPHS_PER_BATCH = 64

# Use GPU for training (if available)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[22]:


def train(model, data):
  model = model.to(DEVICE)

  loss_fn = torch.nn.MSELoss()

  optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)

  data_size = len(data)
  train_loader = DataLoader(
    data[:int(data_size * 0.8)], 
    batch_size=NUM_GRAPHS_PER_BATCH, 
    shuffle=True
  )

  for batch in train_loader:
    # Use GPU
    batch.to(DEVICE)  
    # Reset gradients
    optimizer.zero_grad() 
    # Passing the node features and the edge info
    pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
    # Calculating the loss and gradients
    loss = loss_fn(pred, batch.y)     
    loss.backward()  
    # Update using the gradients
    optimizer.step()   
  return loss, embedding

def train_wrapper():
  print("Starting training...")
  losses = []
  for epoch in range(2000): 
      loss, h = train(MODEL, DATA)
      losses.append(loss)
      if epoch % 100 == 0: 
        print(f"Epoch {epoch} | Train Loss {loss}")
  return losses 

LOSSES = train_wrapper()


# ## Visualize training loss
# 

# In[24]:


import seaborn as sns

def plot_train_loss(): 
  losses_float = [float(loss.cpu().detach().numpy()) for loss in LOSSES] 
  loss_indices = range(len(losses_float))
  ax = sns.lineplot(x=loss_indices, y=losses_float)
  ax.set(xlabel='Epoch', ylabel='Loss')
  
plot_train_loss()


# ## Predict solubility on test data and compare with true solubilities

# In[26]:


import pandas as pd 

def predict(model, data): 
  data_size = len(data)
  test_loader = DataLoader(
    data[int(data_size * 0.8):], 
    batch_size=NUM_GRAPHS_PER_BATCH, 
    shuffle=True
  )

  # Analyze the results for one batch
  test_batch = next(iter(test_loader))

  with torch.no_grad():
    test_batch.to(DEVICE)
    pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
    df = pd.DataFrame()
    df["y_real"] = test_batch.y.tolist()
    df["y_pred"] = pred.tolist()

  df["y_real"] = df["y_real"].apply(lambda row: row[0])
  df["y_pred"] = df["y_pred"].apply(lambda row: row[0])

  axes = sns.scatterplot(data=df, x="y_real", y="y_pred")
  axes.set_xlabel("Real Solubility")
  axes.set_ylabel("Predicted Solubility")

predict(MODEL, DATA)


# In[ ]:




