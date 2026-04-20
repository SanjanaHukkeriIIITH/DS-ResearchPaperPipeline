import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
import json

def build_interactive_graph():
    print("Loading data...")
    try:
        hop1_df = pd.read_parquet("scicite_training_data.parquet")
        hop2_df = pd.read_parquet("hop2_edges.parquet")
    except Exception as e:
        print(f"Error loading parquets: {e}")
        return

    # Filter Hop-1 dataframe to just the seeds we extracted in Hop-2
    sampled_hop1_ids = set(hop2_df["hop1_paper_id"].dropna().unique())
    sampled_hop1_df = hop1_df[hop1_df["citingPaperId"].isin(sampled_hop1_ids)].copy()
    hop0_ids = set(sampled_hop1_df["citedPaperId"].dropna().unique())

    G = nx.DiGraph()
    print("Building Heterogeneous Schema (Papers & Authors)...")

    # 1. Add Hop-0 (Seed) Nodes (VISIBLE)
    for pid in hop0_ids:
        G.add_node(pid,
                   label=pid[:6] + "...", 
                   title=f"<b>HOP-0 (Seed Paper)</b><br>ID: {pid}",
                   color="#2ECC71",  # Green
                   shape="star",
                   size=35,
                   group="seed")

    # 2. Add Hop-1 Nodes & Edges (HIDDEN)
    for _, row in sampled_hop1_df.iterrows():
        h1_id = row["citingPaperId"]
        h0_id = row["citedPaperId"]
        title = row.get("citingPaperTitle", "Unknown Title")
        intent = row.get("label", "Unknown Intent")
        
        G.add_node(h1_id,
                   title=f"<b>HOP-1 Paper</b><br>Title: {title}<br>ID: {h1_id}<br>Intent: {intent}",
                   color="#3498DB",  # Blue
                   shape="square",
                   size=25,          
                   group="hop1",
                   hidden=True)
        G.add_edge(h1_id, h0_id, color="#BDC3C7", title="cites")

    # 3. Add Hop-2 Nodes, Authors, & Edges (HIDDEN)
    # To keep browser fast, we track authors we've already added
    added_authors = set()
    
    for _, row in hop2_df.iterrows():
        h2_id = row["hop2_paper_id"]
        h1_id = row["hop1_paper_id"]
        
        if h2_id not in G:
            title = row.get("hop2_title", "Unknown Title")
            year = row.get("hop2_year", "Unknown Year")
            
            G.add_node(h2_id,
                       title=f"<b>HOP-2 Paper</b><br>Title: {title}<br>Year: {year}<br>ID: {h2_id}",
                       color="#E74C3C",  # Red
                       shape="dot",
                       size=12,
                       group="hop2",
                       hidden=True)
                       
        # Edge: Hop-2 --cites--> Hop-1
        G.add_edge(h2_id, h1_id, color="#F1C40F", title="cites")
        
        # 4. Add Authors & --writes--> Edges (HIDDEN)
        raw_authors = row.get("hop2_author_ids")
        if pd.notna(raw_authors):
            try:
                author_ids = json.loads(raw_authors)
                for aid in author_ids:
                    author_node_id = f"author_{aid}"
                    if author_node_id not in added_authors:
                        G.add_node(author_node_id,
                                   title=f"<b>Author</b><br>ID: {aid}",
                                   color="#9B59B6",  # Purple
                                   shape="triangle",
                                   size=8,
                                   group="author",
                                   hidden=True)
                        added_authors.add(author_node_id)
                    # Edge: Author --writes--> Hop-2 Paper
                    G.add_edge(author_node_id, h2_id, color="#9B59B6", title="writes")
            except:
                pass # JSON decode error on empty

    print(f"Graph built with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")
    print("Generating rich interactive HTML visualization...")
    
    net = Network(height="900px", width="100%", bgcolor="#1E1E1E", font_color="white", directed=True)
    net.from_nx(G)

    # Use BarnesHut for massive graphs, and disable physics after stabilization to save Mac CPU
    net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 1,
        "borderWidthSelected": 2
      },
      "edges": {
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.5 }
        },
        "smooth": { "type": "continuous", "roundness": 0.5 }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.1,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.1
        },
        "minVelocity": 0.75,
        "stabilization": {
          "enabled": true,
          "iterations": 500,
          "updateInterval": 50,
          "onlyDynamicEdges": false,
          "fit": true
        }
      }
    }
    """)

    output_file = "interactive_network.html"
    net.write_html(output_file)
    
    # -------------------------------------------------------------------------
    # POST-PROCESSING: INJECT DRILL-DOWN JAVASCRIPT
    # -------------------------------------------------------------------------
    custom_js = """
    <script type="text/javascript">
        network.on("click", function(params) {
            if (params.nodes.length === 0) return;
            var nodeId = params.nodes[0];
            
            var connectedEdges = network.getConnectedEdges(nodeId);
            var updates = [];
            var clickedGroup = nodes.get(nodeId).group;
            
            connectedEdges.forEach(function(edgeId) {
                var edge = edges.get(edgeId);
                var otherNodeId = (edge.from === nodeId) ? edge.to : edge.from;
                var otherNode = nodes.get(otherNodeId);
                
                if (clickedGroup === 'seed' && otherNode.group === 'hop1') {
                    otherNode.hidden = !otherNode.hidden;
                    updates.push(otherNode);
                } else if (clickedGroup === 'hop1' && (otherNode.group === 'hop2' || otherNode.group === 'author')) {
                    otherNode.hidden = !otherNode.hidden;
                    updates.push(otherNode);
                } else if (clickedGroup === 'hop2' && otherNode.group === 'author') {
                    otherNode.hidden = !otherNode.hidden;
                    updates.push(otherNode);
                }
            });
            
            if (updates.length > 0) {
                nodes.update(updates);
                // Restart physics briefly so nodes can drift apart comfortably
                network.setOptions({ physics: { enabled: true } });
                network.stabilize(50);
            }
        });
    </script>
    </body>
    """
    
    with open(output_file, "r") as f:
        html_content = f.read()
        
    html_content = html_content.replace("</body>", custom_js)
    
    with open(output_file, "w") as f:
        f.write(html_content)
        
    print(f"\\n✅ Advanced Interactive Drill-down Visualization saved as '{output_file}'!")

if __name__ == "__main__":
    build_interactive_graph()
