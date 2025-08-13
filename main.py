import streamlit as st
import requests
import pandas as pd
from collections import defaultdict
import json
from typing import Dict, List, Any, Optional
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Forky Fork Analyzer", layout="wide", page_icon="🔱")

# Define client colors for consistency
CLIENT_COLORS = {
    "prysm": "#FF6B6B",
    "lighthouse": "#4ECDC4",
    "teku": "#45B7D1",
    "nimbus": "#96CEB4",
    "grandine": "#FFEAA7",
    "lodestar": "#DDA0DD",
    "unknown": "#95A5A6"
}

# Cache for frame data to avoid re-fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_frame_cached(base_url: str, frame_id: str) -> Optional[Dict]:
    """Cached version of fetch_frame."""
    url = f"{base_url}/api/v1/frames/{frame_id}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        # Extract the frame from nested structure
        return data.get("data", {}).get("frame")
    except Exception as e:
        return None

@st.cache_data(ttl=300)  # Cache metadata for 5 minutes
def fetch_metadata(base_url: str, slot: int, limit: int = 1000) -> List[Dict]:
    """Fetch metadata for frames at a specific slot."""
    url = f"{base_url}/api/v1/metadata"
    payload = {
        "pagination": {"limit": limit},
        "filter": {"slot": slot}
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("data", {}).get("frames", [])
    except Exception as e:
        st.error(f"Failed to fetch metadata: {e}")
        return []

def fetch_frame(base_url: str, frame_id: str) -> Optional[Dict]:
    """Fetch a specific frame by ID."""
    url = f"{base_url}/api/v1/frames/{frame_id}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        # Extract the frame from nested structure
        return data.get("data", {}).get("frame")
    except Exception as e:
        st.error(f"Failed to fetch frame {frame_id}: {e}")
        return None

def analyze_forks(frames: List[Dict]) -> Dict:
    """Analyze frames to identify fork groups."""
    fork_groups = defaultdict(list)
    block_info = {}
    
    for frame in frames:
        if not frame or not frame.get("data"):
            continue
            
        metadata = frame.get("metadata", {})
        node = metadata.get("node", "unknown")
        consensus_client = metadata.get("consensus_client", "unknown")
        
        # If consensus client is unknown, try to extract from node name
        if consensus_client == "unknown" and node != "unknown":
            # Node names are like "grandine-reth-full-1" or "lighthouse-besu-super-1"
            node_parts = node.split("-")
            if len(node_parts) > 0:
                consensus_client = node_parts[0]
        
        fork_choice = frame.get("data", {})
        nodes = fork_choice.get("fork_choice_nodes", [])
        
        # Find the head block (highest slot with weight)
        head_block = None
        for fc_node in nodes:
            # Convert slot and weight to int for comparison
            node_slot = int(fc_node.get("slot", 0)) if fc_node.get("slot") else 0
            node_weight = int(fc_node.get("weight", 0)) if fc_node.get("weight") else 0
            
            if head_block is None:
                if node_weight > 0:
                    head_block = fc_node
            else:
                head_slot = int(head_block.get("slot", 0)) if head_block.get("slot") else 0
                if node_slot > head_slot and node_weight > 0:
                    head_block = fc_node
        
        if head_block:
            head_root = head_block.get("block_root", "unknown")
            head_slot = int(head_block.get("slot", 0)) if head_block.get("slot") else 0
            
            # Group nodes by their head block
            fork_groups[head_root].append({
                "node": node,
                "consensus_client": consensus_client,
                "head_slot": head_slot,
                "weight": int(head_block.get("weight", 0)) if head_block.get("weight") else 0,
                "validity": head_block.get("validity", "unknown"),
                "justified_epoch": int(fork_choice.get("justified_checkpoint", {}).get("epoch", 0)) if fork_choice.get("justified_checkpoint", {}).get("epoch") else 0,
                "finalized_epoch": int(fork_choice.get("finalized_checkpoint", {}).get("epoch", 0)) if fork_choice.get("finalized_checkpoint", {}).get("epoch") else 0,
            })
            
            # Store block info  
            if head_root not in block_info:
                block_info[head_root] = {
                    "slot": head_slot,
                    "execution_block_hash": head_block.get("execution_block_hash", ""),
                    "parent_root": head_block.get("parent_root", ""),
                }
    
    return {
        "fork_groups": dict(fork_groups),
        "block_info": block_info,
        "num_forks": len(fork_groups),
        "total_nodes": sum(len(nodes) for nodes in fork_groups.values())
    }

def create_fork_tree_visualization(analysis: Dict) -> go.Figure:
    """Create an interactive fork tree visualization."""
    fig = go.Figure()
    
    # Create nodes for each fork
    fork_data = []
    for i, (block_root, nodes) in enumerate(analysis["fork_groups"].items()):
        block_info = analysis["block_info"].get(block_root, {})
        fork_data.append({
            "fork_id": i + 1,
            "block_root": block_root,
            "slot": block_info.get("slot", 0),
            "nodes": nodes,
            "node_count": len(nodes)
        })
    
    # Sort by slot
    fork_data.sort(key=lambda x: x["slot"])
    
    # Create the tree visualization
    for i, fork in enumerate(fork_data):
        x_pos = i
        y_pos = fork["slot"]
        
        # Group nodes by client
        client_counts = defaultdict(int)
        for node in fork["nodes"]:
            client_counts[node["consensus_client"]] += 1
        
        # Create hover text
        hover_text = f"<b>Fork {fork['fork_id']}</b><br>"
        hover_text += f"Block: {fork['block_root'][:16]}...<br>"
        hover_text += f"Slot: {fork['slot']}<br>"
        hover_text += f"Total Nodes: {fork['node_count']}<br><br>"
        hover_text += "<b>Clients:</b><br>"
        for client, count in client_counts.items():
            hover_text += f"{client}: {count}<br>"
        
        # Add the fork as a scatter point
        fig.add_trace(go.Scatter(
            x=[x_pos],
            y=[y_pos],
            mode='markers+text',
            marker=dict(
                size=20 + fork["node_count"] * 5,
                color=fork["slot"],
                colorscale='Viridis',
                showscale=False,
                line=dict(width=2, color='white')
            ),
            text=f"Fork {fork['fork_id']}<br>{fork['node_count']} nodes",
            textposition="top center",
            hovertext=hover_text,
            hoverinfo="text",
            name=f"Fork {fork['fork_id']}"
        ))
    
    fig.update_layout(
        title="Fork Tree Visualization",
        xaxis_title="Fork Index",
        yaxis_title="Slot Number",
        height=500,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_sankey_diagram(analysis: Dict) -> go.Figure:
    """Create a Sankey diagram showing client distribution across forks."""
    
    # Prepare data for Sankey
    clients = set()
    forks = []
    
    for fork_idx, (block_root, nodes) in enumerate(analysis["fork_groups"].items()):
        fork_name = f"Fork {fork_idx + 1}"
        forks.append(fork_name)
        for node in nodes:
            clients.add(node["consensus_client"])
    
    clients = sorted(list(clients))
    
    # Create source, target, and value lists
    source = []
    target = []
    value = []
    labels = clients + forks
    
    for client_idx, client in enumerate(clients):
        for fork_idx, (block_root, nodes) in enumerate(analysis["fork_groups"].items()):
            count = sum(1 for node in nodes if node["consensus_client"] == client)
            if count > 0:
                source.append(client_idx)
                target.append(len(clients) + fork_idx)
                value.append(count)
    
    # Create colors
    node_colors = []
    for label in labels:
        if label in CLIENT_COLORS:
            node_colors.append(CLIENT_COLORS[label])
        elif label.startswith("Fork"):
            node_colors.append("#2C3E50")
        else:
            node_colors.append("#95A5A6")
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color='rgba(0,0,0,0.2)'
        )
    )])
    
    fig.update_layout(
        title="Client Distribution Across Forks",
        height=400,
        font=dict(size=12)
    )
    
    return fig

def create_client_pie_charts(analysis: Dict) -> go.Figure:
    """Create pie charts showing client distribution per fork."""
    
    num_forks = len(analysis["fork_groups"])
    fig = make_subplots(
        rows=1, cols=num_forks,
        subplot_titles=[f"Fork {i+1}" for i in range(num_forks)],
        specs=[[{'type': 'pie'} for _ in range(num_forks)]]
    )
    
    for i, (block_root, nodes) in enumerate(analysis["fork_groups"].items()):
        # Count clients
        client_counts = defaultdict(int)
        for node in nodes:
            client_counts[node["consensus_client"]] += 1
        
        labels = list(client_counts.keys())
        values = list(client_counts.values())
        colors = [CLIENT_COLORS.get(client, "#95A5A6") for client in labels]
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                textinfo='label+value',
                hovertemplate='%{label}: %{value} nodes<br>%{percent}',
                name=f"Fork {i+1}"
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        title="Client Distribution per Fork"
    )
    
    return fig

def create_slot_timeline(analysis: Dict) -> go.Figure:
    """Create a timeline showing slot progression of forks."""
    
    fig = go.Figure()
    
    fork_list = []
    for i, (block_root, nodes) in enumerate(analysis["fork_groups"].items()):
        block_info = analysis["block_info"].get(block_root, {})
        fork_list.append({
            "fork_id": i + 1,
            "slot": block_info.get("slot", 0),
            "node_count": len(nodes),
            "block_root": block_root
        })
    
    fork_list.sort(key=lambda x: x["slot"])
    
    # Create the timeline
    for fork in fork_list:
        fig.add_trace(go.Scatter(
            x=[fork["slot"]],
            y=[fork["fork_id"]],
            mode='markers+text',
            marker=dict(
                size=10 + fork["node_count"] * 3,
                color=fork["fork_id"],
                colorscale='Turbo',
                showscale=False
            ),
            text=f"{fork['node_count']} nodes",
            textposition="top center",
            hovertext=f"Fork {fork['fork_id']}<br>Slot: {fork['slot']}<br>Nodes: {fork['node_count']}<br>Block: {fork['block_root'][:16]}...",
            hoverinfo="text",
            name=f"Fork {fork['fork_id']}"
        ))
    
    fig.update_layout(
        title="Fork Timeline by Slot",
        xaxis_title="Slot Number",
        yaxis_title="Fork ID",
        height=300,
        showlegend=False,
        xaxis=dict(type='linear'),
        yaxis=dict(tickmode='linear', tick0=1, dtick=1)
    )
    
    return fig

def main():
    st.title("🔱 Forky Fork Analyzer")
    st.markdown("*Visualize chain splits and fork divergence from forky data*")
    
    # Initialize session state
    if "forky_url" not in st.session_state:
        st.session_state.forky_url = "https://forky.fusaka-devnet-4.ethpandaops.io"
    if "target_slot" not in st.session_state:
        st.session_state.target_slot = 38197
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        forky_url = st.text_input(
            "Forky URL", 
            value=st.session_state.forky_url,
            help="Base URL of the forky instance"
        )
        st.session_state.forky_url = forky_url
        
        target_slot = st.number_input(
            "Target Slot", 
            value=st.session_state.target_slot,
            min_value=0,
            help="Slot to analyze for forks"
        )
        st.session_state.target_slot = target_slot
        
        st.markdown("---")
        
        analyze_button = st.button("🔍 Analyze Slot", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🎨 Client Colors")
        for client, color in CLIENT_COLORS.items():
            if client != "unknown":
                st.markdown(f"<span style='color: {color}'>● {client}</span>", unsafe_allow_html=True)
    
    # Main content area
    if analyze_button:
        with st.spinner(f"Fetching metadata for slot {target_slot}..."):
            metadata_list = fetch_metadata(forky_url, int(target_slot))
        
        if not metadata_list:
            st.warning(f"No frames found for slot {target_slot}")
            return
        
        # Status message
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ Found {len(metadata_list)} frames for slot {target_slot}")
        with col2:
            with st.expander("💾 Cache Info"):
                st.caption("Frames cached for 1hr")
                st.caption("Metadata cached for 5min")
        
        # Fetch all frames
        frames = []
        progress_bar = st.progress(0, text="Fetching frames...")
        
        for i, meta in enumerate(metadata_list):
            frame_id = meta.get("id")
            if frame_id:
                frame = fetch_frame_cached(forky_url, frame_id)
                if frame:
                    # Add metadata to frame for easy access
                    frame["metadata"] = meta
                    frames.append(frame)
            progress_bar.progress((i + 1) / len(metadata_list), text=f"Fetching frame {i + 1}/{len(metadata_list)}")
        
        progress_bar.empty()
        
        # Analyze forks
        analysis = analyze_forks(frames)
        
        # Display metrics in a nice card layout
        st.markdown("## 📊 Fork Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🌐 Total Nodes", 
                analysis["total_nodes"],
                help="Total number of nodes analyzed"
            )
        
        with col2:
            if analysis["num_forks"] == 1:
                st.metric(
                    "✅ Fork Status", 
                    "Unified",
                    help="All nodes on same fork"
                )
            else:
                st.metric(
                    "⚠️ Fork Count", 
                    analysis["num_forks"],
                    help="Number of different forks detected"
                )
        
        with col3:
            largest = 0
            if analysis["fork_groups"]:
                for nodes in analysis["fork_groups"].values():
                    if len(nodes) > largest:
                        largest = len(nodes)
            st.metric(
                "👥 Largest Fork", 
                largest,
                help="Nodes in the largest fork"
            )
        
        with col4:
            # Calculate health score
            if analysis["num_forks"] == 1:
                health = "🟢 Healthy"
            elif analysis["num_forks"] == 2:
                health = "🟡 Split"
            else:
                health = "🔴 Fragmented"
            st.metric(
                "💚 Network Health", 
                health,
                help="Overall network consensus health"
            )
        
        st.markdown("---")
        
        # Visualizations
        if analysis["num_forks"] > 1:
            st.markdown("## 🎨 Fork Visualizations")
            
            # Tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["🌳 Fork Tree", "🔀 Client Flow", "🥧 Distribution", "⏱️ Timeline"])
            
            with tab1:
                st.plotly_chart(create_fork_tree_visualization(analysis), use_container_width=True)
            
            with tab2:
                st.plotly_chart(create_sankey_diagram(analysis), use_container_width=True)
            
            with tab3:
                st.plotly_chart(create_client_pie_charts(analysis), use_container_width=True)
            
            with tab4:
                st.plotly_chart(create_slot_timeline(analysis), use_container_width=True)
            
            st.markdown("---")
        
        # Detailed Analysis
        st.markdown("## 🔍 Detailed Fork Analysis")
        
        # Create tabs for each fork
        if analysis["fork_groups"]:
            fork_tabs = st.tabs([f"Fork {i+1}" for i in range(len(analysis["fork_groups"]))])
            
            for tab_idx, (tab, (block_root, nodes)) in enumerate(zip(fork_tabs, analysis["fork_groups"].items())):
                with tab:
                    block_info = analysis["block_info"].get(block_root, {})
                    
                    # Fork header
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"### Block `{block_root[:16]}...`")
                        st.caption(f"Full: {block_root}")
                    
                    with col2:
                        st.metric("Slot", block_info.get("slot", "unknown"))
                    
                    with col3:
                        st.metric("Nodes", len(nodes))
                    
                    # Client breakdown
                    client_groups = defaultdict(list)
                    for node in nodes:
                        client_groups[node["consensus_client"]].append(node)
                    
                    st.markdown("#### 🎯 Client Breakdown")
                    client_cols = st.columns(len(client_groups))
                    
                    for col, (client, client_nodes) in zip(client_cols, client_groups.items()):
                        with col:
                            color = CLIENT_COLORS.get(client, "#95A5A6")
                            st.markdown(f"<div style='text-align: center; padding: 10px; background-color: {color}20; border-radius: 10px; border: 2px solid {color};'>"
                                      f"<h4 style='color: {color}; margin: 0;'>{client}</h4>"
                                      f"<h2 style='margin: 0;'>{len(client_nodes)}</h2>"
                                      f"<small>nodes</small></div>", 
                                      unsafe_allow_html=True)
                    
                    # Node details
                    with st.expander("📋 Node Details", expanded=False):
                        df = pd.DataFrame(nodes)
                        df = df[["node", "consensus_client", "weight", "validity", "justified_epoch", "finalized_epoch"]]
                        df["node"] = df["node"].str.split("/").str[-1]  # Simplify node names
                        
                        # Style the dataframe
                        st.dataframe(
                            df.style.background_gradient(subset=['weight'], cmap='YlOrRd'),
                            use_container_width=True,
                            hide_index=True
                        )
        
        # Split Analysis
        if analysis["num_forks"] > 1:
            st.markdown("---")
            st.markdown("## 🔬 Split Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🤝 Client Behavior")
                
                # Check if any client is split across forks
                client_fork_matrix = defaultdict(lambda: defaultdict(int))
                for fork_idx, (block_root, nodes) in enumerate(analysis["fork_groups"].items(), 1):
                    for node in nodes:
                        client = node["consensus_client"]
                        client_fork_matrix[client][f"Fork {fork_idx}"] += 1
                
                split_clients = []
                unified_clients = []
                
                for client in client_fork_matrix:
                    forks_with_client = sum(1 for count in client_fork_matrix[client].values() if count > 0)
                    if forks_with_client > 1:
                        split_clients.append(client)
                    else:
                        unified_clients.append(client)
                
                if split_clients:
                    st.error(f"**Split Clients:** {', '.join(split_clients)}")
                    st.caption("These clients have nodes on multiple forks")
                
                if unified_clients:
                    st.success(f"**Unified Clients:** {', '.join(unified_clients)}")
                    st.caption("These clients are consolidated on single forks")
            
            with col2:
                st.markdown("### 🏛️ Client Alliances")
                
                # Find client groups
                client_forks = defaultdict(set)
                for fork_idx, (block_root, nodes) in enumerate(analysis["fork_groups"].items(), 1):
                    clients_in_fork = set(node["consensus_client"] for node in nodes)
                    for client in clients_in_fork:
                        client_forks[client].add(fork_idx)
                
                fork_alliances = defaultdict(list)
                for client, forks in client_forks.items():
                    fork_key = tuple(sorted(forks))
                    fork_alliances[fork_key].append(client)
                
                for forks, clients in fork_alliances.items():
                    if len(forks) == 1:
                        st.info(f"**Fork {forks[0]}:** {', '.join(clients)}")
                    else:
                        st.warning(f"**Forks {', '.join(map(str, forks))}:** {', '.join(clients)} (split)")

if __name__ == "__main__":
    main()