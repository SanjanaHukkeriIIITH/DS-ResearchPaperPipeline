import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, HeteroConv, SAGEConv
from torch_geometric.utils import softmax

# ==========================================
# 1. The Custom Temporal Attention Layer
# ==========================================
class TemporalGATConv(MessagePassing):
    def __init__(self, node_dim, edge_dim, out_dim):
        # aggr='add' means we sum the weighted messages from all neighbors
        super().__init__(aggr='add', node_dim=0)
        
        # Linear projections for Attention (Queries, Keys, Values)
        self.W_q = nn.Linear(node_dim, out_dim)
        self.W_k = nn.Linear(node_dim, out_dim)
        self.W_v = nn.Linear(node_dim, out_dim)
        self.W_e = nn.Linear(edge_dim, out_dim) # Project the SciBERT edge text
        
        # The Learnable Temporal Decay Parameter (Lambda)
        # We initialize it at 0.1. During backprop, the network learns the true half-life!
        self.lambda_decay = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, edge_index, edge_attr, delta_t):
        # x is the [N, hidden_dim] node embeddings (Abstracts)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # Start message passing. PyG handles routing the variables to the 'message' function
        return self.propagate(edge_index, q=q, k=k, v=v, edge_attr=edge_attr, delta_t=delta_t)

    def message(self, q_i, k_j, v_j, edge_attr, delta_t, index, ptr, size_i):
        # q_i: The citing paper's query vector
        # k_j: The cited paper's key vector
        e_feat = self.W_e(edge_attr)
        
        # Calculate raw semantic attention score
        alpha = (q_i * k_j * e_feat).sum(dim=-1) / math.sqrt(q_i.size(-1))
        
        # --- MODELING TEMPORAL DECAY ---
        # Equation: decay = e^(-lambda * delta_t)
        # We use torch.abs() to mathematically force lambda to remain positive
        decay_factor = torch.exp(-torch.abs(self.lambda_decay) * delta_t.squeeze())
        
        # Penalize the attention score based on the age of the citation
        alpha = alpha * decay_factor
        
        # Normalize scores across the neighborhood so they sum to 1.0
        alpha = softmax(alpha, index, ptr, size_i)
        
        # Save the raw attention weights for tracking and plotting
        self._alpha = alpha.detach()
        
        # Multiply the cited paper's values by the decayed attention score
        return v_j * alpha.unsqueeze(-1)


# ==========================================
# 2. The Full Heterogeneous Network
# ==========================================
class HeteroSemanticModel(nn.Module):
    def __init__(self, paper_dim=768, author_dim=128, edge_dim=768, hidden_dim=64, num_intents=3):
        super().__init__()
        
        self.paper_proj = nn.Linear(paper_dim, hidden_dim)
        self.author_proj = nn.Linear(author_dim, hidden_dim)
        
        self.conv1 = HeteroConv({
            ('author', 'writes', 'paper'): SAGEConv(hidden_dim, hidden_dim),
            ('paper', 'cites', 'paper'): TemporalGATConv(hidden_dim, edge_dim, hidden_dim)
        }, aggr='sum') 
        
        fused_dim = (hidden_dim * 2) + edge_dim + 1 
        
        self.intent_head = nn.Linear(fused_dim, num_intents)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, delta_t_dict):
        x_dict['paper'] = F.relu(self.paper_proj(x_dict['paper']))
        x_dict['author'] = F.relu(self.author_proj(x_dict['author']))
        
        # FIXED: Added parentheses around the tuple keys
        # PyG HeteroData dictionaries use the full edge tuple as the key
        cites_key = ('paper', 'cites', 'paper')
        
        h_dict = self.conv1(
            x_dict, 
            edge_index_dict, 
            edge_attr_dict={cites_key: edge_attr_dict[cites_key]},
            delta_t_dict={cites_key: delta_t_dict[cites_key]}
        )
        
        h_papers = F.relu(h_dict['paper'])
        
        cites_edge_index = edge_index_dict[cites_key]
        src_nodes, dst_nodes = cites_edge_index[0], cites_edge_index[1]
        
        edge_rep = torch.cat([
            h_papers[src_nodes], 
            h_papers[dst_nodes], 
            edge_attr_dict[cites_key], 
            delta_t_dict[cites_key]  
        ], dim=-1)
        
        edge_rep = self.dropout(edge_rep)
        
        intent_logits = self.intent_head(edge_rep)
        
        # Extract the dynamically generated GAT Attention Weights!
        # These are functionally the TRUE Semantic Weights of the citations.
        alpha_scores = None
        for k, v in self.conv1.convs.items():
            if hasattr(v, '_alpha'):
                alpha_scores = v._alpha
                break
        
        return intent_logits, alpha_scores