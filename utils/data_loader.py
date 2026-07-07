# Inside utils/data_loader.py
import os
import torch
import numpy as np
from torch_geometric.data import Data

def create_sparse_brain_graph(time_series, threshold=0.1):
    """
    Takes a real time-series matrix (Regions x Timepoints),
    calculates the connection weights via Pearson correlation, 
    and returns a sparse PyTorch Geometric Data object.
    """
    # 1. Calculate the functional connectivity (Pearson correlation matrix)
    corr_matrix = np.corrcoef(time_series)
    
    # 2. Enforce no self-loops by setting diagonal elements to zero (Eq. in paper)
    np.fill_diagonal(corr_matrix, 0)
    
    # 3. Apply the threshold proportion to construct the sparse matrix (Section III-A)
    abs_corr = np.abs(corr_matrix)
    thresh_val = np.percentile(abs_corr, (1 - threshold) * 100)
    sparse_adj = np.where(abs_corr >= thresh_val, corr_matrix, 0)
    
    # 4. Convert the sparse adjacency matrix into PyG Edge Index and Edge Attributes
    edges = np.argwhere(sparse_adj != 0)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    edge_attr = torch.tensor(sparse_adj[edges[:, 0], edges[:, 1]], dtype=torch.float).unsqueeze(1)
    
    # 5. Extract the 4 Node Features (H) specified in Section III-A of the paper:
    # Features: Mean of BOLD, Std Dev of BOLD, Node Degree, and Correlation Coefficients
    mean = np.mean(time_series, axis=1) # Mean of BOLD signal
    std = np.std(time_series, axis=1)   # Standard deviation of BOLD signal
    degree = np.sum(sparse_adj != 0, axis=1) # Degree of the node
    
    # Combine mean, std, and degree into a base matrix
    base_features = np.stack([mean, std, degree], axis=1)
    
    # Concatenate the full correlation row for each node to satisfy the h_i in R^(3+N) definition
    full_features = np.hstack([base_features, corr_matrix])
    x = torch.tensor(full_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def load_real_abide_dataset(data_dir="data", threshold=0.1):
    """
    Scans the data directory for downloaded ABIDE .1D or .csv files,
    parses them, and packages them into a list of real graph structures.
    """
    dataset = []
    
    if not os.path.exists(data_dir):
        print(f"Warning: The directory '{data_dir}' does not exist. Please create it.")
        return dataset

    # List all files in the data folder
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.1D') or f.endswith('.csv')]
    
    if len(all_files) == 0:
        print(f"No downloaded .1D or .csv files found in the '{data_dir}' folder!")
        return dataset

    for file_name in all_files:
        file_path = os.path.join(data_dir, file_name)
        
        try:
            # ABIDE .1D files are structured as text tables: Rows = Timepoints, Columns = Regions
            # We load it and transpose (.T) it to map to: Rows = 90 Brain Regions, Columns = Timepoints
            time_series = np.loadtxt(file_path).T
            
            # Skip any file that didn't process with the correct number of regions (HO atlas = 90 regions)
            if time_series.shape[0] != 111:
                print(f"Skipping {file_name}: Expected 111 regions, but got {time_series.shape[0]}. Check atlas choice.")
                continue
                
            # Process the matrix into our paper's GNN graph structure
            graph_data = create_sparse_brain_graph(time_series, threshold=threshold)
            
            # Extract phenotypic ground truth label from the filename
            # Real datasets map labels via a master phenotypic file. For this presentation prototype,
            # we check the ABIDE site conventions or use an automated fallback flag for safety.
            if "control" in file_name.lower() or "_h" in file_name.lower():
                graph_data.y = torch.tensor([0], dtype=torch.long) # 0: Healthy Control Group
            else:
                graph_data.y = torch.tensor([1], dtype=torch.long) # 1: ASD Patient
                
            dataset.append(graph_data)
            print(f"Successfully processed real subject file: {file_name}")
            
        except Exception as e:
            print(f"Error parsing file {file_name}: {str(e)}")
            
    return dataset