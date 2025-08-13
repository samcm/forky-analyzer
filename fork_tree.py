from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


def build_fork_tree(frames: List[Dict]) -> Dict:
    """
    Build a proper fork tree from fork choice data.
    Returns information about actual forks (divergent branches), not just different head heights.
    """
    
    # Collect all blocks and their relationships
    block_tree = {}  # block_root -> {slot, parent_root, weight, children, nodes_on_this_head}
    node_heads = {}  # node_name -> head_block_root
    
    for frame in frames:
        if not frame or not frame.get("data"):
            continue
            
        metadata = frame.get("metadata", {})
        node_name = metadata.get("node", "unknown")
        consensus_client = metadata.get("consensus_client", "unknown")
        
        # Parse execution client from node name
        execution_client = "unknown"
        if "-" in node_name:
            parts = node_name.split("-")
            if len(parts) >= 2:
                if consensus_client == "unknown" and parts[0]:
                    consensus_client = parts[0]
                execution_client = parts[1] if parts[1] else "unknown"
        
        fork_choice_nodes = frame.get("data", {}).get("fork_choice_nodes", [])
        if not fork_choice_nodes:
            continue
        
        # Find this node's head (highest weighted block)
        head_block = None
        head_weight = 0
        
        for fc_node in fork_choice_nodes:
            block_root = fc_node.get("block_root")
            parent_root = fc_node.get("parent_root")
            slot = int(fc_node.get("slot", 0))
            weight = int(fc_node.get("weight", 0))
            
            # Add to tree if not exists
            if block_root not in block_tree:
                block_tree[block_root] = {
                    "slot": slot,
                    "parent_root": parent_root,
                    "weight": weight,
                    "children": set(),
                    "nodes_on_this_head": []
                }
            
            # Update weight if this node sees higher weight
            if weight > block_tree[block_root]["weight"]:
                block_tree[block_root]["weight"] = weight
            
            # Track highest weighted block for this node
            if weight > head_weight:
                head_block = block_root
                head_weight = weight
        
        # Record this node's head
        if head_block:
            node_heads[node_name] = head_block
            block_tree[head_block]["nodes_on_this_head"].append({
                "node": node_name,
                "consensus_client": consensus_client,
                "execution_client": execution_client,
                "weight": head_weight
            })
    
    # Build parent-child relationships
    for block_root, block_info in block_tree.items():
        parent = block_info.get("parent_root")
        if parent and parent in block_tree:
            block_tree[parent]["children"].add(block_root)
    
    # Find actual forks (divergence points)
    forks = find_divergent_branches(block_tree, node_heads)
    
    return forks


def find_divergent_branches(block_tree: Dict, node_heads: Dict) -> Dict:
    """
    Find actual forks by identifying divergent branches in the tree.
    Nodes on the same branch (even at different heights) are NOT separate forks.
    """
    
    # Find blocks that have multiple children (fork points)
    fork_points = {}
    for block_root, block_info in block_tree.items():
        if len(block_info["children"]) > 1:
            fork_points[block_root] = block_info
    
    # Group nodes by their actual fork branch
    fork_groups = defaultdict(list)
    fork_weights = {}
    block_info = {}
    
    # For each head block with nodes on it
    heads_with_nodes = {
        block: info for block, info in block_tree.items() 
        if info["nodes_on_this_head"]
    }
    
    # Build ancestry relationships between all heads
    head_relationships = {}  # head -> list of descendant heads
    for head1 in heads_with_nodes:
        head_relationships[head1] = []
        ancestors1 = get_ancestors(head1, block_tree)
        for head2 in heads_with_nodes:
            if head1 != head2 and head1 in get_ancestors(head2, block_tree):
                # head1 is an ancestor of head2
                head_relationships[head1].append(head2)
    
    # Find the "tip" heads (heads that are not ancestors of any other head on the same branch)
    tip_heads = []
    for head in heads_with_nodes:
        if not head_relationships[head]:  # No descendants among the heads
            tip_heads.append(head)
    
    # Group all nodes by their tip head's branch
    fork_id_counter = 0
    processed_heads = set()
    
    for tip_head in tip_heads:
        if tip_head in processed_heads:
            continue
        
        # Collect all nodes on this branch (tip + its ancestors that have nodes)
        branch_nodes = []
        branch_weight = 0
        branch_slot = 0
        
        # Add nodes from the tip
        if tip_head in heads_with_nodes:
            branch_nodes.extend(block_tree[tip_head]["nodes_on_this_head"])
            branch_weight = max(branch_weight, block_tree[tip_head]["weight"])
            branch_slot = max(branch_slot, block_tree[tip_head]["slot"])
            processed_heads.add(tip_head)
        
        # Add nodes from ancestors that are also heads
        for head in heads_with_nodes:
            if head != tip_head and head in get_ancestors(tip_head, block_tree):
                # This head is an ancestor of tip_head, so same branch
                branch_nodes.extend(block_tree[head]["nodes_on_this_head"])
                branch_weight = max(branch_weight, block_tree[head]["weight"])
                processed_heads.add(head)
        
        if branch_nodes:
            fork_key = f"fork_{fork_id_counter}"
            fork_groups[fork_key] = branch_nodes
            fork_weights[fork_key] = branch_weight
            block_info[fork_key] = {
                "slot": branch_slot,
                "root": tip_head
            }
            fork_id_counter += 1
    
    # Sort forks by weight
    sorted_fork_groups = dict(sorted(
        fork_groups.items(),
        key=lambda x: fork_weights.get(x[0], 0),
        reverse=True
    ))
    
    # Rename forks to Fork 1, Fork 2, etc. based on weight order
    final_fork_groups = {}
    final_fork_weights = {}
    final_block_info = {}
    
    for i, (old_key, nodes) in enumerate(sorted_fork_groups.items()):
        new_key = f"fork_{i+1}"
        final_fork_groups[new_key] = nodes
        final_fork_weights[new_key] = fork_weights[old_key]
        final_block_info[new_key] = block_info[old_key]
    
    # Find the actual divergence point if there are multiple forks
    divergence_slot = None
    divergence_block = None
    
    if len(final_fork_groups) > 1:
        # Find the lowest common ancestor of all fork heads
        all_heads = [info["root"] for info in final_block_info.values()]
        divergence_block = find_common_ancestor(all_heads, block_tree)
        if divergence_block and divergence_block in block_tree:
            divergence_slot = block_tree[divergence_block]["slot"]
    
    return {
        "fork_groups": final_fork_groups,
        "block_info": final_block_info,
        "fork_weights": final_fork_weights,
        "num_forks": len(final_fork_groups),
        "total_nodes": sum(len(nodes) for nodes in final_fork_groups.values()),
        "divergence_point": {
            "block": divergence_block,
            "slot": divergence_slot
        } if divergence_block else None
    }


def get_ancestors(block_root: str, block_tree: Dict) -> Set[str]:
    """Get all ancestors of a block."""
    ancestors = set()
    current = block_root
    
    while current in block_tree:
        parent = block_tree[current].get("parent_root")
        if parent and parent != current:  # Avoid infinite loops
            ancestors.add(parent)
            current = parent
        else:
            break
    
    return ancestors


def find_common_ancestor(blocks: List[str], block_tree: Dict) -> Optional[str]:
    """Find the most recent common ancestor of multiple blocks."""
    if not blocks:
        return None
    
    if len(blocks) == 1:
        return None  # No forks if only one head
    
    # Get ancestors for first block (including the block itself)
    common_ancestors = get_ancestors(blocks[0], block_tree)
    common_ancestors.add(blocks[0])  # Include the block itself
    
    # Intersect with ancestors of other blocks
    for block in blocks[1:]:
        block_ancestors = get_ancestors(block, block_tree)
        block_ancestors.add(block)  # Include the block itself
        common_ancestors = common_ancestors.intersection(block_ancestors)
    
    if not common_ancestors:
        return None
    
    # Find the most recent (highest slot) common ancestor
    best_ancestor = None
    best_slot = -1
    
    for ancestor in common_ancestors:
        if ancestor in block_tree:
            slot = block_tree[ancestor]["slot"]
            if slot > best_slot:
                best_slot = slot
                best_ancestor = ancestor
    
    return best_ancestor