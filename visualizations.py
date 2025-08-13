import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Optional
from collections import defaultdict


def format_weight(weight: int) -> str:
    """Format weight in billions/millions for readability."""
    if weight >= 1_000_000_000:
        return f"{weight / 1_000_000_000:.2f}B"
    elif weight >= 1_000_000:
        return f"{weight / 1_000_000:.2f}M"
    elif weight >= 1_000:
        return f"{weight / 1_000:.2f}K"
    else:
        return str(weight)


# Color palette for clients
CLIENT_COLORS = {
    # Consensus clients
    "lighthouse": "#F39C12",  # Changed from red to orange
    "prysm": "#4ECDC4",
    "teku": "#45B7D1",
    "nimbus": "#FFA07A",
    "lodestar": "#98D8C8",
    "grandine": "#B19CD9",
    
    # Execution clients
    "geth": "#627EEA",  # Ethereum blue
    "erigon": "#FF9F1C",  # Orange
    "nethermind": "#2ECC71",  # Green
    "besu": "#9B59B6",  # Purple
    "reth": "#E74C3C",  # Red
}


def create_execution_camp_sankey(analysis: Dict) -> go.Figure:
    """Create a Sankey showing execution camps -> clients -> forks."""
    
    # Define camps
    geth_erigon = {"geth", "erigon"}
    others = {"nethermind", "besu", "reth"}
    
    # Build node labels and indices
    labels = []
    label_to_idx = {}
    idx = 0
    
    # Add camps
    labels.append("geth/erigon")
    label_to_idx["camp_geth_erigon"] = idx
    idx += 1
    
    labels.append("nethermind/besu/reth")
    label_to_idx["camp_others"] = idx
    idx += 1
    
    # Add individual execution clients
    execution_clients = set()
    for nodes in analysis["fork_groups"].values():
        for node in nodes:
            el = node.get("execution_client", "unknown")
            if el not in ["unknown", "nimbusel"]:
                execution_clients.add(el)
    
    execution_clients = sorted(list(execution_clients))
    for el in execution_clients:
        labels.append(el)
        label_to_idx[f"el_{el}"] = idx
        idx += 1
    
    # Add forks with weight info (already sorted by weight)
    forks = []
    fork_positions = []  # Track y positions for proper ordering
    for i, (block_root, nodes) in enumerate(analysis["fork_groups"].items()):
        fork_name = f"Fork {i + 1}"
        weight = analysis["fork_weights"].get(block_root, 0)
        # Format weight for display
        weight_str = format_weight(weight)
        labels.append(f"{fork_name}\n(w: {weight_str})")
        label_to_idx[f"fork_{fork_name}"] = idx
        forks.append(fork_name)
        # Position forks vertically by their order (which is by weight)
        fork_positions.append(i / (len(analysis["fork_groups"]) - 1) if len(analysis["fork_groups"]) > 1 else 0.5)
        idx += 1
    
    # Build links
    source = []
    target = []
    value = []
    link_colors = []
    link_labels = []  # For hover text
    
    # Count flows - use weight instead of node count
    el_to_fork_weights = defaultdict(lambda: defaultdict(int))
    
    for fork_idx, (block_root, nodes) in enumerate(analysis["fork_groups"].items()):
        fork_name = f"Fork {fork_idx + 1}"
        fork_weight = analysis["fork_weights"].get(block_root, 0)
        
        # Count nodes per EL in this fork
        el_counts = defaultdict(int)
        for node in nodes:
            el = node.get("execution_client", "unknown")
            if el not in ["unknown", "nimbusel"]:
                el_counts[el] += 1
        
        # Distribute fork weight proportionally to ELs
        total_el_nodes = sum(el_counts.values())
        for el, count in el_counts.items():
            if total_el_nodes > 0:
                # Weight each EL's contribution by their proportion of nodes
                el_weight = (fork_weight * count) // total_el_nodes
                el_to_fork_weights[el][fork_name] = el_weight
    
    # Camp -> EL links (use sum of weights)
    for el in execution_clients:
        total_weight = sum(el_to_fork_weights[el].values())
        if total_weight > 0:
            if el in geth_erigon:
                source.append(label_to_idx["camp_geth_erigon"])
                target.append(label_to_idx[f"el_{el}"])
                value.append(total_weight)
                link_colors.append("rgba(231, 76, 60, 0.4)")  # Red for geth/erigon
                link_labels.append(f"geth/erigon → {el}: {format_weight(total_weight)}")
            elif el in others:
                source.append(label_to_idx["camp_others"])
                target.append(label_to_idx[f"el_{el}"])
                value.append(total_weight)
                link_colors.append("rgba(52, 152, 219, 0.4)")  # Blue for others
                link_labels.append(f"nethermind/besu/reth → {el}: {format_weight(total_weight)}")
    
    # EL -> Fork links (use weights)
    for el in execution_clients:
        for fork in forks:
            weight = el_to_fork_weights[el][fork]
            if weight > 0:
                source.append(label_to_idx[f"el_{el}"])
                target.append(label_to_idx[f"fork_{fork}"])
                value.append(weight)
                link_labels.append(f"{el} → {fork}: {format_weight(weight)}")
                
                # Color based on whether this is an unexpected split
                if el in geth_erigon:
                    # Check if geth/erigon camp is split
                    geth_erigon_forks = set()
                    for e in geth_erigon:
                        geth_erigon_forks.update(el_to_fork_weights[e].keys())
                    if len(geth_erigon_forks) > 1:
                        link_colors.append("rgba(231, 76, 60, 0.8)")  # Strong red - unexpected!
                    else:
                        link_colors.append("rgba(231, 76, 60, 0.3)")  # Light red - expected
                elif el in others:
                    # Check if others camp is split
                    others_forks = set()
                    for e in others:
                        others_forks.update(el_to_fork_weights[e].keys())
                    if len(others_forks) > 1:
                        link_colors.append("rgba(52, 152, 219, 0.8)")  # Strong blue - unexpected!
                    else:
                        link_colors.append("rgba(52, 152, 219, 0.3)")  # Light blue - expected
                else:
                    link_colors.append("rgba(128, 128, 128, 0.3)")
    
    # Node colors
    node_colors = []
    for label in labels:
        if "geth/erigon" in label:
            node_colors.append("#E74C3C")
        elif "nethermind/besu/reth" in label:
            node_colors.append("#3498DB")
        elif any(el in label for el in ["geth", "erigon"]):
            node_colors.append("#EC7063")
        elif any(el in label for el in ["nethermind", "besu", "reth"]):
            node_colors.append("#5DADE2")
        elif "Fork" in label:
            node_colors.append("#2C3E50")
        else:
            node_colors.append("#95A5A6")
    
    # Calculate y positions for all nodes
    num_camps = 2
    num_els = len(execution_clients)
    num_forks = len(forks)
    
    # Position nodes: camps on left, ELs in middle, forks on right
    y_positions = []
    # Camp positions
    y_positions.extend([0.25, 0.75])  # Two camps
    # EL positions
    for i in range(num_els):
        y_positions.append(i / (num_els - 1) if num_els > 1 else 0.5)
    # Fork positions (already calculated above)
    y_positions.extend(fork_positions)
    
    # X positions: camps at 0.1, ELs at 0.5, forks at 0.9
    x_positions = [0.1] * num_camps + [0.5] * num_els + [0.9] * num_forks
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
            x=x_positions,
            y=y_positions
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
            customdata=link_labels,
            hovertemplate='%{customdata}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title="Execution Client Camps → Forks (Bright colors = unexpected splits!)",
        height=1000,  # Increased height for better visibility
        font=dict(size=11),  # Slightly smaller font to fit more
        margin=dict(t=50, b=25, l=25, r=25)
    )
    
    return fig


def create_sankey_diagram(analysis: Dict, consensus_client: Optional[str] = None) -> go.Figure:
    """Create a Sankey diagram showing execution -> fork flow for a specific consensus client."""
    
    # Collect all unique clients and forks
    execution_clients = set()
    forks = []
    
    # If filtering by consensus client, only include nodes with that client
    filtered_fork_groups = {}
    
    for fork_idx, (block_root, nodes) in enumerate(analysis["fork_groups"].items()):
        fork_name = f"Fork {fork_idx + 1}"
        
        if consensus_client:
            # Filter nodes by consensus client
            filtered_nodes = [n for n in nodes if n["consensus_client"] == consensus_client]
            if filtered_nodes:
                filtered_fork_groups[block_root] = filtered_nodes
                forks.append(fork_name)
                for node in filtered_nodes:
                    execution_clients.add(node.get("execution_client", "unknown"))
        else:
            # Include all nodes
            filtered_fork_groups[block_root] = nodes
            forks.append(fork_name)
            for node in nodes:
                execution_clients.add(node.get("execution_client", "unknown"))
    
    if not filtered_fork_groups:
        return go.Figure()  # Empty figure if no data
    
    execution_clients = sorted(list(execution_clients))
    
    # Create labels for all nodes in order: execution -> forks
    labels = []
    label_to_idx = {}
    idx = 0
    
    # Add execution clients
    for ec in execution_clients:
        labels.append(f"EL: {ec}")
        label_to_idx[f"execution_{ec}"] = idx
        idx += 1
    
    # Add forks
    for fork in forks:
        labels.append(fork)
        label_to_idx[f"fork_{fork}"] = idx
        idx += 1
    
    # Build links
    source = []
    target = []
    value = []
    link_colors = []
    
    # Count execution -> fork combinations
    execution_fork_counts = defaultdict(lambda: defaultdict(int))
    
    for fork_idx, (block_root, nodes) in enumerate(filtered_fork_groups.items()):
        fork_name = f"Fork {fork_idx + 1}"
        for node in nodes:
            ec = node.get("execution_client", "unknown")
            execution_fork_counts[ec][fork_name] += 1
    
    # Add execution -> fork links
    for ec in execution_clients:
        for fork_idx, fork_name in enumerate(forks):
            count = execution_fork_counts[ec][fork_name]
            if count > 0:
                source.append(label_to_idx[f"execution_{ec}"])
                target.append(label_to_idx[f"fork_{fork_name}"])
                value.append(count)
                # Use semi-transparent gray for all links
                link_colors.append("rgba(128, 128, 128, 0.3)")
    
    # Create node colors
    node_colors = []
    for label in labels:
        if label.startswith("EL: "):
            client = label[4:]
            node_colors.append(CLIENT_COLORS.get(client, "#95A5A6"))
        elif label.startswith("Fork"):
            node_colors.append("#2C3E50")
        else:
            node_colors.append("#95A5A6")
    
    title = f"{consensus_client} → Execution → Fork Flow" if consensus_client else "Execution → Fork Flow"
    
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
            color=link_colors
        )
    )])
    
    fig.update_layout(
        title=title,
        height=400,
        font=dict(size=11),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig


def create_fork_composition_treemap(analysis: Dict) -> go.Figure:
    """Create a treemap showing fork composition by CL/EL pairs."""
    
    labels = []
    parents = []
    values = []
    colors = []
    hover_text = []
    
    # Add root
    labels.append("All Forks")
    parents.append("")
    values.append(0)  # Will be sum of children
    colors.append("#2C3E50")
    hover_text.append(f"Total: {analysis['total_nodes']} nodes across {analysis['num_forks']} forks")
    
    # Add each fork and its CL/EL pairs
    for i, (block_root, nodes) in enumerate(analysis["fork_groups"].items()):
        fork_name = f"Fork {i+1}"
        block_info = analysis["block_info"].get(block_root, {})
        
        # Add fork
        labels.append(fork_name)
        parents.append("All Forks")
        values.append(0)  # Will be sum of children
        colors.append("#34495E")
        hover_text.append(
            f"Fork {i+1}\n"
            f"Slot: {block_info.get('slot', 'unknown')}\n"
            f"Nodes: {len(nodes)}\n"
            f"Weight: {format_weight(analysis['fork_weights'].get(block_root, 0))}\n"
            f"Block: {block_root[:16]}..."
        )
        
        # Group by CL/EL pairs
        cl_el_pairs = defaultdict(list)
        for node in nodes:
            cl = node["consensus_client"]
            el = node.get("execution_client", "unknown")
            pair = f"{cl}/{el}"
            cl_el_pairs[pair].append(node)
        
        # Add CL/EL pairs
        for pair, pair_nodes in cl_el_pairs.items():
            cl = pair.split("/")[0]
            labels.append(pair)
            parents.append(fork_name)
            values.append(len(pair_nodes))
            colors.append(CLIENT_COLORS.get(cl, "#95A5A6"))
            
            # Create hover text with node details
            node_names = [n["node"] for n in pair_nodes[:5]]  # Show first 5
            hover = f"{pair}\n"
            hover += f"Count: {len(pair_nodes)}\n"
            hover += f"Weight: {format_weight(pair_nodes[0]['weight'])}\n"
            hover += "Nodes:\n" + "\n".join(f"  - {name}" for name in node_names)
            if len(pair_nodes) > 5:
                hover += f"\n  ... and {len(pair_nodes) - 5} more"
            hover_text.append(hover)
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors),
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        textposition="middle center",
        textfont=dict(size=14)
    ))
    
    fig.update_layout(
        title="Fork Composition by CL/EL Pairs",
        height=600,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig


def create_comprehensive_fork_flow(analysis: Dict) -> tuple[go.Figure, List[Dict]]:
    """Create a comprehensive Sankey showing All Nodes → CL → EL → Forks with detailed breakdowns.
    Returns both the figure and a list of split instances."""
    
    # Build node labels and indices
    labels = []
    label_to_idx = {}
    idx = 0
    
    # Add root node
    labels.append(f"All Nodes\n({analysis['total_nodes']} total)")
    label_to_idx["all_nodes"] = idx
    idx += 1
    
    # Collect unique consensus and execution clients
    consensus_clients = set()
    execution_clients = set()
    
    for nodes in analysis["fork_groups"].values():
        for node in nodes:
            cl = node.get("consensus_client", "unknown")
            el = node.get("execution_client", "unknown")
            if cl != "unknown":
                consensus_clients.add(cl)
            if el not in ["unknown", "nimbusel"]:
                execution_clients.add(el)
    
    consensus_clients = sorted(list(consensus_clients))
    execution_clients = sorted(list(execution_clients))
    
    # Add consensus clients
    for cl in consensus_clients:
        labels.append(f"CL: {cl}")
        label_to_idx[f"cl_{cl}"] = idx
        idx += 1
    
    # Add CL/EL pairs
    cl_el_pairs = set()
    for nodes in analysis["fork_groups"].values():
        for node in nodes:
            cl = node.get("consensus_client", "unknown")
            el = node.get("execution_client", "unknown")
            if cl != "unknown" and el not in ["unknown", "nimbusel"]:
                cl_el_pairs.add((cl, el))
    
    cl_el_pairs = sorted(list(cl_el_pairs))
    for cl, el in cl_el_pairs:
        labels.append(f"{cl}/{el}")
        label_to_idx[f"pair_{cl}_{el}"] = idx
        idx += 1
    
    # Add forks
    fork_labels = []
    for i, (block_root, nodes) in enumerate(analysis["fork_groups"].items()):
        fork_name = f"Fork {i + 1}"
        weight = analysis["fork_weights"].get(block_root, 0)
        node_count = len(nodes)
        block_info = analysis.get("block_info", {}).get(block_root, {})
        slot = block_info.get("slot", "unknown")
        labels.append(f"{fork_name}\nSlot {slot}\n{node_count} nodes\nWeight: {format_weight(weight)}")
        label_to_idx[f"fork_{fork_name}"] = idx
        fork_labels.append(fork_name)
        idx += 1
    
    # Build links
    source = []
    target = []
    value = []
    link_colors = []
    link_labels = []
    
    # Count flows
    cl_counts = defaultdict(int)
    cl_el_counts = defaultdict(lambda: defaultdict(int))
    cl_el_fork_counts = defaultdict(lambda: defaultdict(int))
    cl_el_fork_nodes = defaultdict(lambda: defaultdict(list))  # Track actual node names
    
    for fork_idx, (block_root, nodes) in enumerate(analysis["fork_groups"].items()):
        fork_name = f"Fork {fork_idx + 1}"
        fork_weight = analysis["fork_weights"].get(block_root, 0)
        
        for node in nodes:
            cl = node.get("consensus_client", "unknown")
            el = node.get("execution_client", "unknown")
            node_name = node.get("node", "unknown")
            
            if cl != "unknown":
                cl_counts[cl] += 1
                if el not in ["unknown", "nimbusel"]:
                    cl_el_counts[cl][el] += 1
                    cl_el_fork_counts[(cl, el)][fork_name] += 1
                    cl_el_fork_nodes[(cl, el)][fork_name].append(node_name)
    
    # All Nodes -> CL links
    for cl in consensus_clients:
        if cl_counts[cl] > 0:
            source.append(label_to_idx["all_nodes"])
            target.append(label_to_idx[f"cl_{cl}"])
            value.append(cl_counts[cl])
            link_colors.append(CLIENT_COLORS.get(cl, "rgba(128, 128, 128, 0.3)"))
            link_labels.append(f"All → {cl}: {cl_counts[cl]} nodes")
    
    # CL -> CL/EL pair links
    for cl in consensus_clients:
        for el in execution_clients:
            count = cl_el_counts[cl][el]
            if count > 0:
                source.append(label_to_idx[f"cl_{cl}"])
                target.append(label_to_idx[f"pair_{cl}_{el}"])
                value.append(count)
                link_colors.append(CLIENT_COLORS.get(cl, "rgba(128, 128, 128, 0.3)"))
                link_labels.append(f"{cl} → {cl}/{el}: {count} nodes")
    
    # Track splits for reporting
    split_instances = []
    
    # CL/EL pair -> Fork links
    for (cl, el), fork_counts in cl_el_fork_counts.items():
        if len(fork_counts) > 1:
            # This CL/EL pair is split - record it with node details
            fork_details = {}
            for fork_name in fork_counts:
                # Get fork index to look up block info
                fork_idx = int(fork_name.split()[1]) - 1
                block_root = list(analysis["fork_groups"].keys())[fork_idx]
                block_info = analysis.get("block_info", {}).get(block_root, {})
                weight = analysis.get("fork_weights", {}).get(block_root, 0)
                
                fork_details[fork_name] = {
                    "count": fork_counts[fork_name],
                    "nodes": cl_el_fork_nodes[(cl, el)][fork_name],
                    "slot": block_info.get("slot", "unknown"),
                    "weight": weight
                }
            split_instances.append({
                "cl": cl,
                "el": el,
                "forks": fork_details
            })
        
        for fork_name, count in fork_counts.items():
            if count > 0:
                source.append(label_to_idx[f"pair_{cl}_{el}"])
                target.append(label_to_idx[f"fork_{fork_name}"])
                value.append(count)
                
                # Color based on whether this is expected
                if len(fork_counts) > 1:
                    # This CL/EL pair is split across multiple forks - unexpected!
                    link_colors.append("rgba(231, 76, 60, 0.7)")  # Red
                else:
                    link_colors.append("rgba(52, 152, 219, 0.3)")  # Blue
                
                link_labels.append(f"{cl}/{el} → {fork_name}: {count} nodes")
    
    # Node colors
    node_colors = []
    for label in labels:
        if "All Nodes" in label:
            node_colors.append("#2C3E50")
        elif "CL:" in label:
            cl_name = label.split(":")[1].strip()
            node_colors.append(CLIENT_COLORS.get(cl_name, "#95A5A6"))
        elif "/" in label and "Fork" not in label:
            # CL/EL pair - use CL color
            cl_name = label.split("/")[0]
            node_colors.append(CLIENT_COLORS.get(cl_name, "#95A5A6"))
        elif "Fork" in label:
            node_colors.append("#34495E")
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
            color=link_colors,
            customdata=link_labels,
            hovertemplate='%{customdata}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title="Complete Fork Flow: All Nodes → Consensus → CL/EL Pairs → Forks (Red = Unexpected Splits)",
        height=1200,
        font=dict(size=10),
        margin=dict(t=50, b=25, l=25, r=25)
    )
    
    return fig, split_instances


def create_fork_tree_sankey(analysis: Dict, frames: List[Dict]) -> go.Figure:
    """Create a Sankey diagram showing fork divergence from common ancestors."""
    
    # Find common ancestor blocks by analyzing fork choice data
    fork_heads = analysis["fork_groups"]
    
    # For simplicity, find the most recent split point
    # This would be the last block that appears in multiple fork choice trees
    common_blocks = {}
    
    for frame in frames:
        if not frame or not frame.get("data"):
            continue
        
        fc_nodes = frame.get("data", {}).get("fork_choice_nodes", [])
        for node in fc_nodes:
            block_root = node.get("block_root")
            slot = int(node.get("slot", 0))
            if block_root:
                if block_root not in common_blocks:
                    common_blocks[block_root] = {"slot": slot, "count": 0}
                common_blocks[block_root]["count"] += 1
    
    # Find blocks that appear in many nodes (potential common ancestors)
    threshold = len(frames) * 0.7  # Block should appear in 70% of nodes
    potential_ancestors = [
        (root, info) for root, info in common_blocks.items() 
        if info["count"] >= threshold
    ]
    
    # Sort by slot to find the most recent common ancestor
    potential_ancestors.sort(key=lambda x: x[1]["slot"], reverse=True)
    
    # Find the split point (where forks diverge)
    recent_split = None
    split_slot = 0
    
    if potential_ancestors:
        # The most recent widely-shared block
        recent_split = potential_ancestors[0][0]
        split_slot = potential_ancestors[0][1]["slot"]
    
    # Build Sankey diagram
    labels = []
    label_to_idx = {}
    idx = 0
    
    # Add common ancestor if found
    if recent_split:
        labels.append(f"Common Ancestor\nSlot {split_slot}")
        label_to_idx[recent_split] = idx
        idx += 1
    
    # Add fork heads (simplified labels with just essential info)
    fork_order = []
    for i, (head_root, nodes) in enumerate(fork_heads.items()):
        fork_order.append(head_root)
        head_info = analysis["block_info"][head_root]
        head_slot = head_info["slot"]
        node_count = len(nodes)
        weight = analysis["fork_weights"].get(head_root, 0)
        
        # Simplified label
        label = f"Fork {i+1}\n"
        label += f"Slot {head_slot}\n"
        label += f"{node_count} nodes\n"
        label += f"Weight: {format_weight(weight)}"
        
        labels.append(label)
        label_to_idx[head_root] = idx
        idx += 1
    
    # Build links
    source = []
    target = []
    value = []
    link_colors = []
    
    if recent_split and recent_split in label_to_idx:
        # Link from common ancestor to each fork head
        for head_root in fork_heads.keys():
            if head_root in label_to_idx:
                source.append(label_to_idx[recent_split])
                target.append(label_to_idx[head_root])
                # Use weight instead of node count
                value.append(analysis["fork_weights"].get(head_root, 0))
                
                # Color based on fork size
                if len(fork_heads[head_root]) > len(fork_heads) / 2:
                    link_colors.append("rgba(46, 204, 113, 0.5)")  # Green for majority
                else:
                    link_colors.append("rgba(231, 76, 60, 0.5)")  # Red for minority
    
    # Create hover text and node colors
    node_colors = []
    hover_texts = []
    
    for i, label in enumerate(labels):
        if "Common Ancestor" in label:
            node_colors.append("#3498DB")  # Blue for ancestor
            hover_texts.append(label.replace('\n', '<br>'))
        elif "Fork" in label:
            # Extract fork index
            lines = label.split('\n')
            fork_idx = int(lines[0].split()[1]) - 1  # Extract fork number
            
            if fork_idx < len(fork_order):
                head_root = fork_order[fork_idx]
                nodes = fork_heads[head_root]
                head_info = analysis["block_info"][head_root]
                weight = analysis["fork_weights"].get(head_root, 0)
                
                # Color based on node count
                count = len(nodes)
                if count > 20:
                    node_colors.append("#2ECC71")  # Green for large fork
                elif count > 10:
                    node_colors.append("#F39C12")  # Orange for medium fork
                else:
                    node_colors.append("#E74C3C")  # Red for small fork
                
                # Build hover text with all details
                hover = f"<b>Fork {fork_idx + 1}</b><br>"
                hover += f"Slot: {head_info['slot']} (+{head_info['slot'] - split_slot if split_slot else 0} since split)<br>"
                hover += f"Block: {head_root[:16]}...<br>"
                hover += f"Total: {len(nodes)} nodes<br>"
                hover += f"Weight: {format_weight(weight)}<br>"
                hover += "<br><b>CL/EL Breakdown:</b><br>"
                
                # Get CL/EL pairs
                cl_el_pairs = defaultdict(int)
                for node in nodes:
                    cl = node["consensus_client"]
                    el = node.get("execution_client", "unknown")
                    cl_el_pairs[f"{cl}/{el}"] += 1
                
                for pair, pair_count in sorted(cl_el_pairs.items(), key=lambda x: x[1], reverse=True):
                    hover += f"{pair}: {pair_count}<br>"
                
                hover_texts.append(hover)
            else:
                node_colors.append("#95A5A6")
                hover_texts.append(label.replace('\n', '<br>'))
    
    if not source:
        # No common ancestor found - just show the fork heads
        st.info("No common ancestor found in recent history. Showing fork heads only.")
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
            customdata=hover_texts,
            hovertemplate='%{customdata}<extra></extra>'
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
            customdata=[format_weight(v) for v in value],
            hovertemplate='Weight: %{customdata}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title="Fork Tree Structure",
        height=400,
        font=dict(size=12),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig