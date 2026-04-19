import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn import Linear, Dropout

class SemanticEdgeClassifier(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_classes):
        super().__init__()
        # 1. Node Embedding Layer 
        # (Upgrades our dummy '1' features into learnable representations)
        self.node_emb = Linear(node_in_dim, hidden_dim)
        
        # 2. Graph Convolutional Layers (GraphSAGE)
        # (These allow nodes to learn from their neighbors' topology)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        # 3. The Edge Predictor MLP
        # It takes: Source Node (hidden_dim) + Target Node (hidden_dim) + SciBERT text (edge_in_dim)
        classifier_input_dim = (hidden_dim * 2) + edge_in_dim
        
        self.classifier_hidden = Linear(classifier_input_dim, hidden_dim)
        self.classifier_out = Linear(hidden_dim, num_classes)
        self.dropout = Dropout(0.5)

    def forward(self, x, edge_index, edge_attr):
        # --- Step 1: Message Passing (Learning the Topology) ---
        x = self.node_emb(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # --- Step 2: Edge Feature Construction ---
        # Extract the node embeddings for the citing (src) and cited (dst) papers
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        x_src = x[src_nodes]
        x_dst = x[dst_nodes]
        
        # Concatenate the topology features with your 768-dimensional SciBERT text features
        edge_representation = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        
        # --- Step 3: Intent Classification ---
        h = F.relu(self.classifier_hidden(edge_representation))
        h = self.dropout(h)
        out = self.classifier_out(h)
        
        return out

# Instantiate the model to verify the architecture
model = SemanticEdgeClassifier(
    node_in_dim=1,          # Our dummy node features are size 1
    edge_in_dim=768,        # SciBERT embeddings are size 768
    hidden_dim=64,          # The size of our learned node embeddings
    num_classes=3           # Background (0), Method (1), Result (2)
)

print(model)