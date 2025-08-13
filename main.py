import streamlit as st
import requests
import json
from typing import Dict, List, Optional

from analysis import analyze_forks
from typing import Dict
from visualizations import (
    create_fork_composition_treemap,
    create_fork_tree_sankey,
    create_sankey_diagram,
    create_comprehensive_fork_flow,
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

@st.cache_data(ttl=3600)  # 1 hour - analysis results can be cached longer
def analyze_forks_cached(frames: str) -> Dict:
    """Cached version of analyze_forks that works with serializable data."""
    frames_list = json.loads(frames)
    return analyze_forks(frames_list)

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
        
        # Analyze forks (using cached version)
        frames_json = json.dumps(frames)
        analysis = analyze_forks_cached(frames_json)
        
        # Filter out small forks
        def filter_small_forks(analysis: Dict, min_size: int) -> Dict:
            """Filter out forks with fewer than min_size nodes."""
            filtered_groups = {}
            filtered_weights = {}
            
            for block_root, nodes in analysis["fork_groups"].items():
                if len(nodes) >= min_size:
                    filtered_groups[block_root] = nodes
                    filtered_weights[block_root] = analysis["fork_weights"].get(block_root, 0)
            
            return {
                "fork_groups": filtered_groups,
                "block_info": analysis["block_info"],
                "fork_weights": filtered_weights,
                "num_forks": len(filtered_groups),
                "total_nodes": sum(len(nodes) for nodes in filtered_groups.values())
            }
        
        # Apply fork size filter
        filtered_analysis = filter_small_forks(analysis, min_fork_size)
        
        # Show filtering info if any forks were filtered
        if analysis["num_forks"] != filtered_analysis["num_forks"]:
            st.info(f"Filtered {analysis['num_forks'] - filtered_analysis['num_forks']} small forks (< {min_fork_size} nodes)")
        
        # Use filtered analysis for visualizations
        analysis = filtered_analysis
        
        # Visualizations
        if analysis["num_forks"] > 1:
            # Add tabs for different views
            view_tabs = st.tabs(["🔍 Comprehensive Overview", "📊 Fork Analysis", "🔀 Client Flows"])
            
            with view_tabs[0]:
                st.markdown("### 🔍 Complete Fork Flow Overview")
                fig, split_instances = create_comprehensive_fork_flow(analysis)
                
                # Show split instances at the top if any
                if split_instances:
                    st.error(f"🚨 **Unexpected Splits Detected** - {len(split_instances)} CL/EL pair{'s' if len(split_instances) > 1 else ''} split across multiple forks")
                    
                    for split in split_instances:
                        with st.expander(f"⚠️ **{split['cl']}/{split['el']}** - Split across {len(split['forks'])} forks", expanded=True):
                            for fork_name, fork_data in split['forks'].items():
                                st.markdown(f"**{fork_name}** - Slot {fork_data['slot']} - Weight: {format_weight(fork_data['weight'])} ({fork_data['count']} node{'s' if fork_data['count'] > 1 else ''}):")
                                for node_name in fork_data['nodes']:
                                    st.markdown(f"  • `{node_name}`")
                    st.markdown("---")
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
                # Consensus client specific flows
                st.markdown("### 🔀 Consensus → Execution → Fork Flow")
                
                # Get all consensus clients
                consensus_clients = set()
                for nodes in analysis["fork_groups"].values():
                    for node in nodes:
                        consensus_clients.add(node["consensus_client"])
                consensus_clients = sorted(list(consensus_clients))
                
                # Create tabs for each consensus client
                if len(consensus_clients) > 1:
                    tabs = st.tabs([f"🟢 {cl}" for cl in consensus_clients])
                    
                    for i, cl in enumerate(consensus_clients):
                        with tabs[i]:
                            fig = create_sankey_diagram(analysis, cl)
                            if fig.data:  # Only show if there's data
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Count nodes for this CL
                                cl_nodes = sum(
                                    len([n for n in nodes if n["consensus_client"] == cl])
                                    for nodes in analysis["fork_groups"].values()
                                )
                                st.caption(f"Shows how {cl} nodes ({cl_nodes} total) pair with execution clients and distribute across forks")
                            else:
                                st.info(f"No data for {cl}")
                else:
                    # Only one consensus client, show directly
                    st.plotly_chart(create_sankey_diagram(analysis), use_container_width=True)
                    st.caption("Shows how nodes pair with execution clients and which forks they end up on")
            
        elif analysis["num_forks"] == 1:
            st.success("✅ Network is in consensus - no forks detected!")
        else:
            st.warning("No data to analyze")

if __name__ == "__main__":
    main()