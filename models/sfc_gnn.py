import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool

class RaGCNLayer(MessagePassing):
    """Brain region perception-based graph convolution layer (Section III-B1)."""
    def __init__(self, in_channels, out_channels, num_regions=90):
        super(RaGCNLayer, self).__init__(aggr='add')
        self.phi = nn.PReLU() 
        self.gamma = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.gamma, 0.1)
        
        # Each brain region gets its own dedicated linear transformation matrix[cite: 1]
        self.region_transforms = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(num_regions)
        ])

    def forward(self, x, edge_index, edge_attr):
        # Step 1: Apply the unique region-based transformation to each node[cite: 1]
        x_transformed = torch.stack([self.region_transforms[i](x[i]) for i in range(x.size(0))], dim=0)
        
        # Step 2: Propagate the messages across neighbor connections[cite: 1]
        out = self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr)
        
        # Step 3: Combine self-information and neighbor information (Eq. 3)[cite: 1]
        return self.phi((1 + self.gamma) * x_transformed + out)

    def message(self, x_j, edge_attr):
        # Scale neighbor features by the connectivity edge weights[cite: 1]
        return edge_attr * x_j


class SFC_GNN(nn.Module):
    """
    Structure Feature Combined Graph Neural Network (Section III-B4)[cite: 1].
    """
    def __init__(self, num_regions=90, in_features=3):
        super(SFC_GNN, self).__init__()
        # 1. Graph Convolution Layer
        self.conv1 = RaGCNLayer(in_features, 16, num_regions=num_regions)
        
        # Learnable mapping vector 'm' for feature-based scoring (Eq. 4)
        self.pooling_vector = nn.Parameter(torch.randn(16, 1)) 
        
        # Final Multilayer Perceptron Classifier
        self.classifier = nn.Sequential(
            nn.Linear(16 * 2, 16), 
            nn.PReLU(),
            nn.Linear(16, 2)       
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Step A: Run Graph Convolution[cite: 1]
        x = self.conv1(x, edge_index, edge_attr)
        
        # Step B: GSF Pooling Layer Self-Attention Scores (Eq. 4)[cite: 1]
        scores = torch.sigmoid(torch.matmul(x, self.pooling_vector)).squeeze()
        
        # Step C: Readout Layer Splicing (Eq. 9)[cite: 1]
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_representation = torch.cat([x_mean, x_max], dim=1) 
        
        # Step D: Predict Final Label[cite: 1]
        out = self.classifier(graph_representation)
        
        return out, scores