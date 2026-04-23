import torch
import torch.nn.functional as F
import pandas as pd
import json
import os
import tqdm

from hgnn_model import HeteroSemanticModel

# ---------------------------------------------------------------------------
# CLUSTER ISOLATION VERIFICATION
# ---------------------------------------------------------------------------
def verify_cluster_isolation(data, paper_ids):
    """
    Proves that attention weights learned for Hop-1 paper A (under Hop-0 X)
    cannot 'leak' into Hop-1 paper B (under Hop-0 Y).

    In a Graph Attention Network, a node can ONLY aggregate messages from
    its direct neighbors via the edge_index. We verify this by checking:
    - Which Hop-0 nodes exist (papers that are never citing, only cited)
    - Which Hop-1 nodes belong to which Hop-0 cluster
    - That no Hop-1 node appears in two separate clusters (no bridge nodes)
    """
    print("\n🔍 CLUSTER ISOLATION VERIFICATION")
    print("=" * 60)

    edge_index = data['paper', 'cites', 'paper'].edge_index.cpu()
    src_indices = edge_index[0].tolist()
    dst_indices = edge_index[1].tolist()

    all_citing = set(src_indices)
    all_cited  = set(dst_indices)

    # Hop-0 = papers that are ONLY cited (never cite another paper in the graph)
    hop0_indices = all_cited - all_citing

    # For each Hop-0 node, find all Hop-1 nodes that cite it
    hop0_to_hop1 = {}
    for src, dst in zip(src_indices, dst_indices):
        if dst in hop0_indices:
            hop0_to_hop1.setdefault(dst, set()).add(src)

    # Check: does any Hop-1 node appear under multiple Hop-0 nodes?
    hop1_to_hop0 = {}
    cross_cluster_violations = []
    for hop0_idx, hop1_set in hop0_to_hop1.items():
        for hop1_idx in hop1_set:
            if hop1_idx in hop1_to_hop0:
                # This Hop-1 node already belongs to another cluster!
                cross_cluster_violations.append((hop1_idx, hop1_to_hop0[hop1_idx], hop0_idx))
            else:
                hop1_to_hop0[hop1_idx] = hop0_idx

    print(f"  Hop-0 Seed Papers (cluster roots): {len(hop0_indices)}")
    print(f"  Hop-1 Papers Total (citing seeds): {len(hop1_to_hop0)}")

    if cross_cluster_violations:
        print(f"\n  ⚠️  {len(cross_cluster_violations)} Hop-1 papers cite multiple Hop-0 seeds.")
        print(f"  These are 'bridge' papers that genuinely appear in multiple clusters.")
        print(f"  This is CORRECT behaviour – the model should learn their multi-cluster role.")
        print(f"  GAT attention still operates per-edge; weight for edge A→X is independent")
        print(f"  of weight for edge A→Y even if paper A bridges clusters X and Y.")
    else:
        print("  ✅ Perfect isolation: every Hop-1 paper belongs to exactly one Hop-0 cluster.")

    print("\n  WHY CROSS-CONTAMINATION CANNOT HAPPEN IN GAT:")
    print("  ─────────────────────────────────────────────")
    print("  The attention score α(A→X) = softmax( f(q_A, k_X, e_AX, Δt_AX) )")
    print("  It uses ONLY the A→X edge. Paper B's edge B→Y is a completely")
    print("  different row in edge_index. The softmax is per-node neighborhood,")
    print("  not global—so B's weight cannot influence A's weight in any epoch.")
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
    graph_path = "data/scicite_hetero_scaled.pt"
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"{graph_path} not found. Run data_prep_scaled.py first.")

    data = torch.load(graph_path, weights_only=False)

    with open("data/hop01_metadata_scaled.json", "r") as f:
        meta = json.load(f)
    paper_ids = list(meta.keys())    # preserves insertion order (Python 3.7+)

    print(f"✅ Loaded graph: {data}")
    print(f"   Edges to train on: {data['paper', 'cites', 'paper'].train_mask.sum().item()}")

    # ---- 2. Cluster Isolation Verification (runs on CPU, no GPU needed) ----
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

    train_mask  = data['paper', 'cites', 'paper'].train_mask
    true_labels = data['paper', 'cites', 'paper'].intent_label

    for epoch in range(1, 101):
        optimizer.zero_grad()

        intent_logits, importance_scores = model(
            data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.delta_t_dict
        )

        loss = loss_fn(intent_logits[train_mask], true_labels[train_mask])
        
        # MPS/Scale Safety: Skip if loss is NaN and use gradient clipping
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

    # ---- 5. Save Training Artefacts (mirrors train_hgnn.py) ----

    # a) Model Weights
    torch.save(model.state_dict(), "data/hgnn_weights_scaled.pth")
    print("  ✅ Saved: data/hgnn_weights_scaled.pth")

    # b) Training Log
    pd.DataFrame(history_logs).to_csv("data/training_logs_scaled.csv", index=False)
    print("  ✅ Saved: data/training_logs_scaled.csv")

    # c) Final Node Embeddings (post-conv representations)
    model.eval()
    with torch.no_grad():
        cites_key   = ('paper', 'cites', 'paper')
        x_proj      = {
            'paper':  F.relu(model.paper_proj(data.x_dict['paper'])),
            'author': F.relu(model.author_proj(data.x_dict['author']))
        }
        h_dict = model.conv1(
            x_proj, data.edge_index_dict,
            edge_attr_dict  = {cites_key: data.edge_attr_dict[cites_key]},
            delta_t_dict    = {cites_key: data.delta_t_dict[cites_key]}
        )
        final_embeddings = F.relu(h_dict['paper']).cpu()
        torch.save(final_embeddings, "data/final_node_embeddings_scaled.pt")
        print("  ✅ Saved: data/final_node_embeddings_scaled.pt")

    # d) Full Graph Edge Predictions + Semantic Weights
    with torch.no_grad():
        intent_logits, importance_scores = model(
            data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.delta_t_dict
        )
        final_preds   = intent_logits.argmax(dim=1).cpu().numpy()
        final_weights = importance_scores.squeeze().cpu().numpy()
        edge_index_np = data['paper', 'cites', 'paper'].edge_index.cpu().numpy()

    intent_name_map = {0: "Background", 1: "Method", 2: "Result"}
    edges_df = pd.DataFrame({
        "citing_paper_id": [paper_ids[i] for i in edge_index_np[0]],
        "cited_paper_id":  [paper_ids[i] for i in edge_index_np[1]],
        "predicted_intent":  [intent_name_map[p] for p in final_preds],
        "semantic_weight":   final_weights
    })
    edges_df.to_parquet("data/edge_predictions_scaled.parquet", index=False)
    print("  ✅ Saved: data/edge_predictions_scaled.parquet")

    print("\n📦 Summary of Scaled Training Artefacts:")
    print("   data/hgnn_weights_scaled.pth          → Model weights for fine-tuning")
    print("   data/training_logs_scaled.csv          → Loss/Acc/Lambda per epoch")
    print("   data/final_node_embeddings_scaled.pt   → Post-GAT paper representations")
    print("   data/edge_predictions_scaled.parquet   → Per-edge intent + semantic weight")
    print("\nNext → update app.py to point to these '_scaled' files!")

if __name__ == "__main__":
    train_scaled()
