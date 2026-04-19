import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear, Dropout

class SemanticDualHeadGAT(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_intents):
        super().__init__()
        # 1. Project node features (We will use abstract embeddings here soon)
        self.node_emb = Linear(node_in_dim, hidden_dim)
        
        # 2. Graph Attention Layers (GATv2 allows edge features to influence attention!)
        # heads=4 means multi-head attention, allowing it to look for multiple textual patterns simultaneously
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, heads=4, concat=False)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, heads=4, concat=False)
        
        # 3. Fused Edge Representation
        edge_fused_dim = (hidden_dim * 2) + edge_in_dim
        
        # 4. The Dual Heads
        # Head A: Classifies the intent (Method, Result, Background)
        self.intent_head = Linear(edge_fused_dim, num_intents)
        
        # Head B: Predicts continuous importance (Probability of being a Key Citation)
        self.importance_head = Linear(edge_fused_dim, 1)
        
        self.dropout = Dropout(0.5)

    def forward(self, x, edge_index, edge_attr):
        # --- Step 1: Topological Attention (Message Passing) ---
        x = self.node_emb(x)
        x = F.relu(self.gat1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.gat2(x, edge_index, edge_attr=edge_attr))
        
        # --- Step 2: Edge Feature Extraction ---
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        # Combine the attention-aware nodes with the SciBERT text vector
        edge_rep = torch.cat([x[src_nodes], x[dst_nodes], edge_attr], dim=-1)
        edge_rep = self.dropout(F.relu(edge_rep))
        
        # --- Step 3: Dual Output ---
        # Logits for intent classification (e.g., [0.2, 8.5, 1.1])
        intent_logits = self.intent_head(edge_rep)
        
        # Sigmoid squashes the importance score to a continuous value between 0.0 and 1.0
        importance_score = torch.sigmoid(self.importance_head(edge_rep))
        
        return intent_logits, importance_score

# Instantiate the new architecture
model = SemanticDualHeadGAT(
    node_in_dim=1,       # Placeholder until we encode the abstracts
    edge_in_dim=768,     # SciBERT embeddings
    hidden_dim=64, 
    num_intents=3
)
print(model)