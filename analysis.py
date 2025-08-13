from typing import Dict, List
from collections import defaultdict


def analyze_forks(frames: List[Dict]) -> Dict:
    """Analyze frames to identify fork groups."""
    
    # Group nodes by their head block
    fork_groups = defaultdict(list)
    block_info = {}  # Store block metadata
    
    for frame in frames:
        if not frame or not frame.get("data"):
            continue
            
        metadata = frame.get("metadata", {})
        node_name = metadata.get("node", "unknown")
        consensus_client = metadata.get("consensus_client", "unknown")
        
        # Parse clients from node name if needed (format: consensus-execution-type-number)
        if "-" in node_name:
            parts = node_name.split("-")
            if len(parts) >= 2:
                # If consensus client is unknown, try to parse from node name
                if consensus_client == "unknown" and parts[0]:
                    consensus_client = parts[0]
                # Parse execution client from second part
                execution_client = parts[1] if parts[1] else "unknown"
        else:
            execution_client = "unknown"
        
        # Find the head block (highest slot with non-zero weight)
        fork_choice_nodes = frame.get("data", {}).get("fork_choice_nodes", [])
        if not fork_choice_nodes:
            continue
            
        # Sort by slot descending to find the head
        sorted_nodes = sorted(
            fork_choice_nodes,
            key=lambda x: (int(x.get("slot", 0)), int(x.get("weight", 0))),
            reverse=True
        )
        
        # Find the first node with non-zero weight (the head)
        head_block = None
        head_slot = 0
        head_weight = 0
        
        for node in sorted_nodes:
            weight = int(node.get("weight", 0))
            if weight > 0:
                head_block = node.get("block_root")
                head_slot = int(node.get("slot", 0))
                head_weight = weight
                break
        
        if head_block:
            fork_groups[head_block].append({
                "node": node_name,
                "consensus_client": consensus_client,
                "execution_client": execution_client,
                "slot": head_slot,
                "weight": head_weight
            })
            
            # Store block info if we haven't seen it
            if head_block not in block_info:
                block_info[head_block] = {
                    "slot": head_slot,
                    "root": head_block
                }
    
    # Calculate total weight for each fork first
    fork_weights = {}
    for block_root, nodes in fork_groups.items():
        # Use the max weight from any node (they should be the same for the same block)
        fork_weights[block_root] = max(node["weight"] for node in nodes) if nodes else 0
    
    # Sort fork groups by weight (largest first)
    sorted_groups = dict(sorted(
        fork_groups.items(),
        key=lambda x: fork_weights[x[0]],
        reverse=True
    ))
    
    return {
        "fork_groups": sorted_groups,
        "block_info": block_info,
        "fork_weights": fork_weights,
        "num_forks": len(sorted_groups),
        "total_nodes": sum(len(nodes) for nodes in fork_groups.values())
    }