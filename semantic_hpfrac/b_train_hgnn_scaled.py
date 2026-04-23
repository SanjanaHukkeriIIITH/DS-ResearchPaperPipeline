import torch
import torch.nn.functional as F
import pandas as pd
import json
import os

from hgnn_model import HeteroSemanticModel

# ---------------------------------------------------------------------------
# CLUSTER ISOLATION VERIFICATION
# ---------------------------------------------------------------------------
def verify_cluster_isolation(data, paper_ids):
    print("\n🔍 CLUSTER ISOLATION VERIFICATION")
    print("=" * 60)

    edge_index = data['paper', 'cites', 'paper'].edge_index.cpu()
    src_indices = edge_index[0].tolist()
    dst_indices = edge_index[1].tolist()

    all_citing = set(src_indices)
    all_cited  = set(dst_indices)

    hop0_indices = all_cited - all_citing

    hop0_to_hop1 = {}
    for src, dst in zip(src_indices, dst_indices):
        if dst in hop0_indices:
            hop0_to_hop1.setdefault(dst, set()).add(src)

    hop1_to_hop0 = {}
    cross_cluster_violations = []
    for hop0_idx, hop1_set in hop0_to_hop1.items():
        for hop1_idx in hop1_set:
            if hop1_idx in hop1_to_hop0:
                cross_cluster_violations.append((hop1_idx, hop1_to_hop0[hop1_idx], hop0_idx))
            else:
                hop1_to_hop0[hop1_idx] = hop0_idx

    print(f"  Hop-0 Seed Papers (cluster roots): {len(hop0_indices)}")
    print(f"  Hop-1 Papers Total (citing seeds): {len(hop1_to_hop0)}")

    if cross_cluster_violations:
        print(f"\n  ⚠️  {len(cross_cluster_violations)} Hop-1 papers cite multiple Hop-0 seeds.")
        print(f"  These are 'bridge' papers that genuinely appear in multiple clusters.")
    else:
        print("  ✅ Perfect isolation: every Hop-1 paper belongs to exactly one Hop-0 cluster.")
    print("=" * 60)

# ---------------------------------------------------------------------------
# MAIN TRAINING ROUTINE
# ---------------------------------------------------------------------------
def train_scaled():
    device = torch.device(
        'mps'  if torch.backends.mps.is_available()  else
        'cuda' if torch.cuda.is_available()           else
        'cpu'
    )
    print(f"🚀 Training on device: {device}")

    # ---- 1. Load Graph ----
    graph_path = "data/data_v2/scicite_hetero_scaled.pt"
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"{graph_path} not found. Run data_prep_scaled.py first.")

    data = torch.load(graph_path, weights_only=False)

    with open("data/data_v2/hop01_metadata_scaled.json", "r") as f:
        meta = json.load(f)
    paper_ids = list(meta.keys())

    print(f"✅ Loaded graph: {data}")
    print(f"   Edges to train on: {data['paper', 'cites', 'paper'].train_mask.sum().item()}")

    # ---- 2. Cluster Isolation Verification ----
    verify_cluster_isolation(data, paper_ids)

    data = data.to(device)

    # ---- 3. Model ----
    model = HeteroSemanticModel(
        paper_dim  = 768,
        author_dim = 128,
        edge_dim   = 768,
        hidden_dim = 64,
        num_intents= 3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn   = torch.nn.CrossEntropyLoss()

    # ---- 4. Training Loop ----
    print("\n🚀 Commencing HGNN Phase 3 Training (100 Epochs)...")
    model.train()
    history_logs = []

    cites_key   = ('paper', 'cites', 'paper')
    train_mask  = data[cites_key].train_mask
    true_labels = data[cites_key].intent_label

    for epoch in range(1, 101):
        optimizer.zero_grad()

        intent_logits, importance_scores = model(
            data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.delta_t_dict
        )

        loss = loss_fn(intent_logits[train_mask], true_labels[train_mask])
        
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            print(f"⚠️ Warning: NaN loss at epoch {epoch}, skipping backward pass.")
            optimizer.zero_grad()

        with torch.no_grad():
            preds = intent_logits[train_mask].argmax(dim=1)
            acc   = (preds == true_labels[train_mask]).float().mean().item()

            lambda_val = 0.1
            for k, v in model.conv1.convs.items():
                if hasattr(v, 'lambda_decay'):
                    lambda_val = v.lambda_decay.item()
                    break

            history_logs.append({
                "epoch":    epoch,
                "loss":     loss.item(),
                "accuracy": acc,
                "lambda":   lambda_val
            })

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                      f"Accuracy: {acc*100:.1f}% | Learned Λ: {lambda_val:.4f}")

    print("\n🎉 Training Complete! Exporting Models & Analytics...")

    # ---- 5. Save Training Artefacts ----
    torch.save(model.state_dict(), "data/data_v2/hgnn_weights_scaled.pth")
    pd.DataFrame(history_logs).to_csv("data/data_v2/training_logs_scaled.csv", index=False)

    model.eval()
    with torch.no_grad():
        x_proj = {
            'paper':  F.relu(model.paper_proj(data.x_dict['paper'])),
            'author': F.relu(model.author_proj(data.x_dict['author']))
        }
        h_dict = model.conv1(
            x_proj, data.edge_index_dict,
            edge_attr_dict  = {cites_key: data.edge_attr_dict[cites_key]},
            delta_t_dict    = {cites_key: data.delta_t_dict[cites_key]}
        )
        final_embeddings = F.relu(h_dict['paper']).cpu()
        torch.save(final_embeddings, "data/data_v2/final_node_embeddings_scaled.pt")

    # ---- 6. Export Edge Predictions for Metrics Processing ----
    with torch.no_grad():
        intent_logits, importance_scores = model(
            data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.delta_t_dict
        )
        final_preds   = intent_logits.argmax(dim=1).cpu().numpy()
        final_weights = importance_scores.squeeze().cpu().numpy()
        edge_index_np = data[cites_key].edge_index.cpu().numpy()
        delta_t_np    = data.delta_t_dict[cites_key].cpu().numpy().squeeze()

    intent_name_map = {0: "Background", 1: "Method", 2: "Result"}
    
    edges_df = pd.DataFrame({
        "citing_paper_idx": edge_index_np[0],
        "cited_paper_idx":  edge_index_np[1],
        "citing_paper_id":  [paper_ids[i] for i in edge_index_np[0]],
        "cited_paper_id":   [paper_ids[i] for i in edge_index_np[1]],
        "predicted_intent": [intent_name_map[p] for p in final_preds],
        "raw_importance":   final_weights,
        "delta_t":          delta_t_np
    })
    
    edges_df.to_parquet("data/data_v2/edge_predictions_scaled.parquet", index=False)
    print("  ✅ Saved: data/data_v2/edge_predictions_scaled.parquet")
    print("\nNext → run compute_metrics.py to calculate True 2-Hop Semantic hp-frac!")

if __name__ == "__main__":
    train_scaled()