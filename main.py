import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, List, Optional

from fork_tree import build_fork_tree
from visualizations import (
    create_fork_composition_treemap,
    create_fork_tree_sankey,
    create_comprehensive_fork_flow,
    create_node_fork_table,
    format_weight
)

# Cache configuration
st.set_page_config(
    page_title="🔱 Forky Fork Analyzer",
    page_icon="🔱",
    layout="wide"
)

@st.cache_data(ttl=21600)  # 6 hours - frame data is immutable by ID
def fetch_frame_cached(base_url: str, frame_id: str) -> Optional[Dict]:
    """Cached version of fetch_frame."""
    return fetch_frame(base_url, frame_id)


@st.cache_data(ttl=600)  # 10 minutes - metadata changes more frequently
def fetch_metadata(base_url: str, slot: int, limit: int = 1000) -> List[Dict]:
    """Fetch metadata for frames at a specific slot."""
    try:
        response = requests.post(
            f"{base_url}/api/v1/metadata",
            json={
                "pagination": {"limit": limit},
                "filter": {"slot": slot}
            }
        )
        response.raise_for_status()
        data = response.json()
        # Handle nested structure
        frames = data.get("data", {}).get("frames", [])
        return frames
    except Exception as e:
        st.error(f"Failed to fetch metadata: {e}")
        return []

def fetch_frame(base_url: str, frame_id: str) -> Optional[Dict]:
    """Fetch a specific frame by ID."""
    try:
        response = requests.get(f"{base_url}/api/v1/frames/{frame_id}")
        response.raise_for_status()
        data = response.json()
        # Handle nested structure
        frame = data.get("data", {}).get("frame", {})
        return frame
    except Exception as e:
        st.error(f"Failed to fetch frame {frame_id}: {e}")
        return None


def main():
    st.title("🔱 Forky Fork Analyzer")
    st.caption("Analyze Ethereum network forks and client distributions")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        base_url = st.text_input(
            "Forky API URL",
            value="https://forky.fusaka-devnet-4.ethpandaops.io",
            help="Base URL of the Forky API"
        )
        
        slot_number = st.number_input(
            "Slot Number",
            min_value=0,
            value=38197,
            step=1,
            help="Slot number to analyze"
        )
        
        st.markdown("---")
        
        min_fork_size = st.number_input(
            "Minimum Fork Size",
            min_value=1,
            value=1,
            step=1,
            help="Hide forks with fewer nodes than this"
        )
        
        st.markdown("---")
        
        analyze_button = st.button(f"📊 Analyze Slot {slot_number}", type="primary", use_container_width=True)
        
        if st.button("🔄 Refresh Cache", type="secondary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.caption("Built with Streamlit & Plotly")
    
    # Main content
    if analyze_button:
        with st.spinner(f"Fetching frames for slot {slot_number}..."):
            metadata = fetch_metadata(base_url, slot_number)
        
        if not metadata:
            st.error("No frames found for this slot")
            return
        
        st.success(f"Found {len(metadata)} frames at slot {slot_number}")
        
        # Fetch all frames
        frames = []
        progress = st.progress(0)
        
        for i, meta in enumerate(metadata):
            frame_id = meta.get("id")
            if frame_id:
                frame = fetch_frame_cached(base_url, frame_id)
                if frame:
                    frames.append(frame)
            progress.progress((i + 1) / len(metadata))
        
        progress.empty()
        
        # Analyze forks using proper tree-based detection
        with st.spinner("Building fork tree..."):
            analysis = build_fork_tree(frames)
        
        # Filter out small forks
        def filter_small_forks(analysis: Dict, min_size: int) -> Dict:
            """Filter out forks with fewer than min_size nodes."""
            filtered_groups = {}
            filtered_weights = {}
            filtered_block_info = {}
            
            for fork_key, nodes in analysis["fork_groups"].items():
                if len(nodes) >= min_size:
                    filtered_groups[fork_key] = nodes
                    filtered_weights[fork_key] = analysis["fork_weights"].get(fork_key, 0)
                    filtered_block_info[fork_key] = analysis["block_info"].get(fork_key, {})
            
            return {
                "fork_groups": filtered_groups,
                "block_info": filtered_block_info,
                "fork_weights": filtered_weights,
                "num_forks": len(filtered_groups),
                "total_nodes": sum(len(nodes) for nodes in filtered_groups.values()),
                "divergence_point": analysis.get("divergence_point")
            }
        
        # Apply fork size filter
        filtered_analysis = filter_small_forks(analysis, min_fork_size)
        
        # Show filtering info if any forks were filtered
        if analysis["num_forks"] != filtered_analysis["num_forks"]:
            st.info(f"Filtered {analysis['num_forks'] - filtered_analysis['num_forks']} small forks (< {min_fork_size} nodes)")
        
        # Use filtered analysis for visualizations
        analysis = filtered_analysis
        
        # Show divergence point if found
        if analysis.get("divergence_point") and analysis["divergence_point"]["slot"]:
            st.info(f"🔀 **Fork divergence detected at slot {analysis['divergence_point']['slot']}** - Block: `{analysis['divergence_point']['block'][:16]}...`")
        
        # Visualizations
        if analysis["num_forks"] > 1:
            # Add tabs for different views
            view_tabs = st.tabs(["🔍 Comprehensive Overview", "📊 Fork Analysis", "📋 Node Table"])
            
            with view_tabs[0]:
                st.markdown("### 🔍 Complete Fork Flow Overview")
                fig, split_instances = create_comprehensive_fork_flow(analysis)
                
                # Show split instances at the top if any
                if split_instances:
                    st.error(f"🚨 **{len(split_instances)} Unexpected Split{'s' if len(split_instances) > 1 else ''}**")
                    
                    # Compact display in columns
                    cols = st.columns(min(len(split_instances), 3))
                    for i, split in enumerate(split_instances):
                        with cols[i % 3]:
                            st.markdown(f"**{split['cl']}/{split['el']}**")
                            for fork_name, fork_data in split['forks'].items():
                                st.caption(f"{fork_name}: `{', '.join(fork_data['nodes'])}`")
                else:
                    st.success("✅ No unexpected splits detected - all CL/EL pairs are on the same fork")
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Single pane view: Shows how all nodes flow through consensus clients, pair with execution clients, and end up on different forks. Red links indicate unexpected splits where the same CL/EL pair ends up on different forks.")
            
            with view_tabs[1]:
                # Fork visualization options
                viz_tab1, viz_tab2 = st.tabs(["🌲 Fork Divergence", "🗂️ Fork Composition"])
                
                with viz_tab1:
                    st.plotly_chart(create_fork_tree_sankey(analysis, frames), use_container_width=True)
                    st.caption("Fork divergence showing common ancestor and fork weights")
                
                with viz_tab2:
                    st.plotly_chart(create_fork_composition_treemap(analysis), use_container_width=True)
                    st.caption("Fork composition by CL/EL pairs. Size = node count, Color = consensus client")
            
            with view_tabs[2]:
                # Node table view
                st.markdown("### 📋 Node Fork Assignment Table")
                
                # Create summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Nodes", analysis["total_nodes"])
                with col2:
                    st.metric("Total Forks", analysis["num_forks"])
                with col3:
                    largest_fork = max(analysis["fork_groups"].values(), key=len)
                    st.metric("Largest Fork", f"{len(largest_fork)} nodes")
                with col4:
                    # Show fork distribution
                    fork_sizes = [len(nodes) for nodes in analysis["fork_groups"].values()]
                    st.metric("Fork Distribution", f"{'/'.join(map(str, fork_sizes))}")
                
                # Get dataframe and colors
                df, fork_colors = create_node_fork_table(analysis)
                
                # Display color legend
                st.markdown("**Fork Colors:**")
                cols = st.columns(min(len(fork_colors), 6))
                for i, (fork_name, color) in enumerate(fork_colors.items()):
                    if f"Fork {i+1}" in df['Fork'].values:
                        with cols[i % 6]:
                            st.markdown(f"<span style='background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px;'>{fork_name}</span>", unsafe_allow_html=True)
                
                # Display the dataframe with highlighting
                st.dataframe(
                    df.style.apply(lambda row: [f'background-color: {fork_colors.get(row["Fork"], "#34495E")}20' for _ in row], axis=1),
                    use_container_width=True,
                    height=600
                )
                
                st.caption("Table shows all nodes with their consensus and execution clients. Rows are colored by fork assignment.")
            
        elif analysis["num_forks"] == 1:
            st.success("✅ Network is in consensus - no forks detected!")
        else:
            st.warning("No data to analyze")

if __name__ == "__main__":
    main()