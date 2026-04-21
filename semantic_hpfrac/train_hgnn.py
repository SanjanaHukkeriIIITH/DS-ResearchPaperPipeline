import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import pandas as pd
import json
import numpy as np

# Import your beautiful existing model
from hgnn_model import HeteroSemanticModel

def build_hetero_dataset():
    print("Loading Graph Metadata & Embeddings...")
    
    # 1. Load Data
    hop1_df = pd.read_parquet("scicite_training_data.parquet")
    hop2_df = pd.read_parquet("hop2_edges.parquet")
    
    embeddings = torch.load("node_embeddings.pt", map_location="cpu", weights_only=True)
    
    with open("hop01_metadata.json", "r") as f:
        meta01 = json.load(f)
        
    print(f"Loaded {len(embeddings)} SciBERT vectors.")

    data = HeteroData()
    
    # 2. Build Paper Node Mapping
    # Only keep papers for which we have an embedding
    paper_ids = list(embeddings.keys())
    paper2idx = {pid: i for i, pid in enumerate(paper_ids)}
    num_papers = len(paper_ids)
    
    # Stack the embeddings to create x_dict['paper']
    x_paper = torch.zeros(num_papers, 768)
    for pid, idx in paper2idx.items():
        x_paper[idx] = embeddings[pid]
    data['paper'].x = x_paper
    
    # 3. Build Author Node Mapping
    all_authors = set()
    for raw_authors in hop2_df["hop2_author_ids"].dropna():
        try:
            author_ids = json.loads(raw_authors)
            for aid in author_ids: all_authors.add(aid)
        except:
            pass
            
    author2idx = {aid: i for i, aid in enumerate(list(all_authors))}
    # Initialize random embeddings for authors, these will be updated via GraphSAGE!
    data['author'].x = torch.randn(len(all_authors), 128)
    
    # 4. Construct Edges
    # Edge Type 1: ('paper', 'cites', 'paper')
    cites_src, cites_dst, edge_attrs, delta_ts, intent_labels, label_masks = [], [], [], [], [], []
    
    intent_map = {"background": 0, "method": 1, "result": 2}
    
    # Hop-1 -> Hop-0 Edges
    active_hop1_ids = set(hop2_df["hop1_paper_id"].dropna().unique())
    sampled_hop1_df = hop1_df[hop1_df["citingPaperId"].isin(active_hop1_ids)]
    
    for _, row in sampled_hop1_df.iterrows():
        src = row["citingPaperId"]
        dst = row["citedPaperId"]
        
        if src in paper2idx and dst in paper2idx:
            cites_src.append(paper2idx[src])
            cites_dst.append(paper2idx[dst])
            
            # Semantic weight vector for the edge: we use the Citing Paper's vector
            # since it physically contains the embedded text of the citation sentence!
            edge_attrs.append(embeddings[src])
            
            # Intent Label (Target for training)
            label_str = row.get("label", "")
            if label_str in intent_map:
                intent_labels.append(intent_map[label_str])
                label_masks.append(True) # This is a labeled edge we can train on!
            else:
                intent_labels.append(-1)
                label_masks.append(False)
                
            # Delta Time computation
            src_year = meta01.get(src, {}).get("year", 0)
            dst_year = meta01.get(dst, {}).get("year", 0)
            try:
                dt = max(0, int(src_year) - int(dst_year))
            except:
                dt = 0
            delta_ts.append(dt)

    # Hop-2 -> Hop-1 Edges
    for _, row in hop2_df.iterrows():
        src = row["hop2_paper_id"]
        dst = row["hop1_paper_id"]
        
        if src in paper2idx and dst in paper2idx:
            cites_src.append(paper2idx[src])
            cites_dst.append(paper2idx[dst])
            
            edge_attrs.append(embeddings[src])
            intent_labels.append(-1)
            label_masks.append(False) # No ground truth labels for Hop-2 in Scicite
            
            # Delta Time
            src_year = row.get("hop2_year")
            dst_year = meta01.get(dst, {}).get("year", 0)
            try:
                dt = max(0, int(float(src_year)) - int(float(dst_year)))
            except:
                dt = 0
            delta_ts.append(dt)

    # Convert to Tensors
    data['paper', 'cites', 'paper'].edge_index = torch.tensor([cites_src, cites_dst], dtype=torch.long)
    data['paper', 'cites', 'paper'].edge_attr = torch.stack(edge_attrs)
    # We pass delta_t as a [num_edges, 1] tensor as required by TemporalGATConv
    data['paper', 'cites', 'paper'].delta_t = torch.tensor(delta_ts, dtype=torch.float).unsqueeze(1)
    
    # Store labels and masks for training
    data['paper', 'cites', 'paper'].intent_label = torch.tensor(intent_labels, dtype=torch.long)
    data['paper', 'cites', 'paper'].train_mask = torch.tensor(label_masks, dtype=torch.bool)
    
    # Edge Type 2: ('author', 'writes', 'paper')
    writes_src, writes_dst = [], []
    for _, row in hop2_df.iterrows():
        pid = row["hop2_paper_id"]
        if pid in paper2idx:
            raw_authors = row.get("hop2_author_ids")
            if pd.notna(raw_authors):
                try:
                    for aid in json.loads(raw_authors):
                        if aid in author2idx:
                            writes_src.append(author2idx[aid])
                            writes_dst.append(paper2idx[pid])
                except: pass
                
    if writes_src:
        data['author', 'writes', 'paper'].edge_index = torch.tensor([writes_src, writes_dst], dtype=torch.long)
    else:
        data['author', 'writes', 'paper'].edge_index = torch.empty((2, 0), dtype=torch.long)

    return data, list(paper2idx.keys())

def train():
    data, paper_ids = build_hetero_dataset()
    print(f"\n✅ PyG HeteroData Schema Assembled:")
    print(data)
    print(f"Edges to Train Intent On: {data['paper', 'cites', 'paper'].train_mask.sum().item()}\n")

    # Initialize your amazing Model!
    model = HeteroSemanticModel(
        paper_dim=768, 
        author_dim=128, 
        edge_dim=768, 
        hidden_dim=64, 
        num_intents=3
    )
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("\n🚀 Commencing HGNN Phase 3 Training (100 Epochs)...")
    model.train()
    history_logs = []
    
    for epoch in range(1, 101):
        optimizer.zero_grad()
        
        intent_logits, importance_scores = model(
            data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.delta_t_dict
        )
        
        train_mask = data['paper', 'cites', 'paper'].train_mask
        true_labels = data['paper', 'cites', 'paper'].intent_label
        
        # 3. Task Loss: Intent Classification (Cross Entropy)
        # We rely on the GAT Alpha Softmax mechanism to learn the optimal semantic distribution 
        # naturally, bypassing BERT's anisotropic embedding constraints!
        loss = loss_fn(intent_logits[train_mask], true_labels[train_mask])
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            preds = intent_logits[train_mask].argmax(dim=1)
            acc = (preds == true_labels[train_mask]).sum().item() / max(1, train_mask.sum().item())
            
            lambda_val = 0.1
            for k, v in model.conv1.convs.items():
                if hasattr(v, 'lambda_decay'):
                    lambda_val = v.lambda_decay.item()
                    break
                    
            history_logs.append({"epoch": epoch, "loss": loss.item(), "accuracy": acc, "lambda": lambda_val})
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Accuracy: {acc*100:.1f}% | Learned Λ: {lambda_val:.4f}")

    print("\n🎉 Training Complete! Exporting Models and Visual Analytics to Disk...")
    
    # 1. Save Weights
    torch.save(model.state_dict(), "hgnn_weights.pth")
    # 2. Save Log History
    pd.DataFrame(history_logs).to_csv("training_logs.csv", index=False)
    
    # 3. Predict final graph attributes
    model.eval()
    with torch.no_grad():
        cites_key = ('paper', 'cites', 'paper')
        x_dict_fwd = {'paper': F.relu(model.paper_proj(data.x_dict['paper'])), 
                      'author': F.relu(model.author_proj(data.x_dict['author']))}
        h_dict = model.conv1(x_dict_fwd, data.edge_index_dict, 
                             edge_attr_dict={cites_key: data.edge_attr_dict[cites_key]}, 
                             delta_t_dict={cites_key: data.delta_t_dict[cites_key]})
        
        final_node_embeddings = F.relu(h_dict['paper']).cpu()
        torch.save(final_node_embeddings, "final_node_embeddings.pt")
        
        intent_logits, importance_scores = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.delta_t_dict)
        final_preds = intent_logits.argmax(dim=1).cpu().numpy()
        final_weights = importance_scores.squeeze().cpu().numpy()
        edge_index_np = data['paper', 'cites', 'paper'].edge_index.cpu().numpy()
        
        # Format mapping dict
        intent_name_map = {0: "Background", 1: "Method", 2: "Result", -1: "Unknown"}
        mapped_intents = [intent_name_map.get(lbl, "Unknown") for lbl in final_preds]
        
        edges_df = pd.DataFrame({
            "citing_paper_id": [paper_ids[i] for i in edge_index_np[0]],
            "cited_paper_id": [paper_ids[i] for i in edge_index_np[1]],
            "predicted_intent": mapped_intents,
            "semantic_weight": final_weights
        })
        edges_df.to_parquet("edge_predictions.parquet", index=False)
        
    print("✅ Successfully saved `hgnn_weights.pth`, `training_logs.csv`, `final_node_embeddings.pt`, and `edge_predictions.parquet`!")

if __name__ == "__main__":
    train()
