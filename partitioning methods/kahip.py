#!/usr/bin/env python
# kahip.py - KaHIP Knowledge Graph Partitioning for ECML PKDD 2026
# 
# This implementation uses the Karlsruhe High Quality Partitioning (KaHIP) library
# for high-quality graph partitioning with multiple configuration options.

import os
import torch
import numpy as np
from collections import defaultdict
import dgl
import json
import time
import subprocess
import tempfile
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import from project config
try:
    from config import DATASET_CONFIGS, get_partition_dir
except ImportError:
    # Fallback if config not available
    DATASET_CONFIGS = {
        "FB15K": {"num_entities": 14951, "num_relations": 1345, "data_dir": "data/FB15K"},
        "FB15K-237": {"num_entities": 14541, "num_relations": 237, "data_dir": "data/FB15K-237"},
        "YAGO3-10": {"num_entities": 123182, "num_relations": 37, "data_dir": "data/YAGO3-10"}
    }
    
    def get_partition_dir(method, dataset, num_parts):
        return f"data/partitions_{method}/{dataset}_{num_parts}parts"


def load_dict(file_path):
    """Load dictionary file with format: id\tvalue"""
    loaded_dict = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        uid, val = parts[0], parts[1]
                        loaded_dict[val] = int(uid)
    except FileNotFoundError:
        print(f"Warning: Dictionary file not found: {file_path}")
        return {}
    return loaded_dict


def read_triplets(file_path, entity2id, relation2id):
    """Read triplets file and convert to numeric IDs"""
    triplets = []
    line_count = 0
    skipped = 0
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line_count += 1
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        head, relation, tail = parts[0], parts[1], parts[2]
                        if head in entity2id and relation in relation2id and tail in entity2id:
                            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))
                        else:
                            skipped += 1
                    else:
                        skipped += 1
    except FileNotFoundError:
        print(f"Error: Triplets file not found: {file_path}")
        return np.array([])
    
    print(f"Processed {line_count} lines, found {len(triplets)} valid triplets (skipped {skipped})")
    return np.array(triplets)


class PartitionQualityMetrics:
    """Metrics for evaluating partition quality"""
    
    @staticmethod
    def calculate_edge_cut_ratio(partitions, triplets, entity_to_partition):
        """Calculate edge cut ratio - percentage of edges crossing partitions"""
        if len(triplets) == 0:
            return 0.0
        
        edge_cuts = 0
        for h, r, t in triplets:
            h_part = entity_to_partition.get(h)
            t_part = entity_to_partition.get(t)
            if h_part is not None and t_part is not None and h_part != t_part:
                edge_cuts += 1
        
        return edge_cuts / len(triplets)
    
    @staticmethod
    def calculate_balance_factor(partitions):
        """Calculate balance factor metrics"""
        partition_sizes = [len(nodes) for nodes in partitions.values()]
        
        if not partition_sizes:
            return {
                'balance_factor': 1.0,
                'max_size': 0,
                'min_size': 0,
                'avg_size': 0,
                'std_dev': 0,
                'imbalance_ratio': 1.0
            }
        
        avg_size = np.mean(partition_sizes)
        max_size = max(partition_sizes)
        min_size = min(partition_sizes)
        std_dev = np.std(partition_sizes)
        
        if avg_size > 0:
            balance_factor = min_size / max_size
            imbalance_ratio = max_size / avg_size
        else:
            balance_factor = 1.0
            imbalance_ratio = 1.0
        
        return {
            'balance_factor': float(balance_factor),
            'max_size': int(max_size),
            'min_size': int(min_size),
            'avg_size': float(avg_size),
            'std_dev': float(std_dev),
            'imbalance_ratio': float(imbalance_ratio)
        }


def check_kahip_installation():
    """Check if KaHIP is installed and available"""
    try:
        # Try to run KaHIP's tool with --help
        result = subprocess.run(['kaffpa', '--help'], 
                               capture_output=True, 
                               text=True, 
                               timeout=5)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            # Try alternative executable name
            result = subprocess.run(['KaHIP/kaffpa', '--help'],
                                   capture_output=True,
                                   text=True,
                                   timeout=5)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Warning: KaHIP not found. Will use fallback random partitioning.")
            return False


def convert_to_metis_format(triplets, entity2id, relation2id, output_file):
    """
    Convert knowledge graph to METIS format for KaHIP
    
    METIS format:
    - First line: <num_vertices> <num_edges> <fmt> <ncon>
    - Following lines: adjacency list
    """
    # Get all unique entities
    all_entities = list(set([h for h, _, _ in triplets] + [t for _, _, t in triplets]))
    entity_to_idx = {entity: i for i, entity in enumerate(all_entities)}
    
    num_vertices = len(all_entities)
    
    # Build adjacency list (undirected, no self-loops, no duplicate edges)
    adjacency_dict = defaultdict(set)
    
    for h, r, t in triplets:
        h_idx = entity_to_idx[h]
        t_idx = entity_to_idx[t]
        
        if h_idx != t_idx:  # Avoid self-loops
            # Add undirected edge
            adjacency_dict[h_idx].add(t_idx)
            adjacency_dict[t_idx].add(h_idx)
    
    # Count unique undirected edges
    num_edges = sum(len(neighbors) for neighbors in adjacency_dict.values()) // 2
    
    # Write METIS format file
    with open(output_file, 'w') as f:
        # Header: vertices edges fmt ncon
        # fmt=0 means no weights, ncon=1 means one balance constraint
        f.write(f"{num_vertices} {num_edges} 0 1\n")
        
        # Write adjacency lists
        for i in range(num_vertices):
            neighbors = sorted(list(adjacency_dict[i]))
            if neighbors:
                f.write(" ".join(str(n+1) for n in neighbors) + "\n")  # METIS uses 1-based indexing
            else:
                f.write("\n")  # Empty line for isolated vertices
    
    return num_vertices, num_edges, all_entities, entity_to_idx


def parse_kahip_output(output_file, num_vertices, all_entities):
    """
    Parse KaHIP output file to extract partition assignments
    """
    partitions = defaultdict(set)
    entity_to_partition = {}
    
    try:
        with open(output_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_vertices:
                    break
                line = line.strip()
                if line:
                    part_id = int(line)
                    entity = all_entities[i]
                    partitions[part_id].add(entity)
                    entity_to_partition[entity] = part_id
    except Exception as e:
        print(f"Error parsing KaHIP output: {e}")
        return {}, {}
    
    return dict(partitions), entity_to_partition


def partition_knowledge_graph_random(triplets, entity2id, relation2id, num_parts):
    """
    Fallback method: Random partitioning
    """
    print(f"\n{'='*50}")
    print(f"Random Knowledge Graph Partitioning (Fallback)")
    print(f"{'='*50}")
    
    # Get all unique entities
    all_entities = list(set([h for h, _, _ in triplets] + [t for _, _, t in triplets]))
    
    # Randomly assign entities to partitions
    partitions = defaultdict(set)
    entity_to_partition = {}
    
    for entity in all_entities:
        part_id = np.random.randint(0, num_parts)
        partitions[part_id].add(entity)
        entity_to_partition[entity] = part_id
    
    # Count edge cuts
    edge_cuts = 0
    for h, r, t in triplets:
        if entity_to_partition.get(h) != entity_to_partition.get(t):
            edge_cuts += 1
    
    print(f"Random partitioning created {len(partitions)} partitions with {edge_cuts} edge cuts")
    
    return dict(partitions), entity_to_partition, edge_cuts


def partition_knowledge_graph_kahip(triplets, entity2id, relation2id, num_parts, 
                                    mode='strong', imbalance=3):
    """
    Partition the knowledge graph using KaHIP (Karlsruhe High Quality Partitioning)
    
    Args:
        triplets: Array of triplets
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        num_parts: Number of partitions
        mode: Partitioning mode ('fast', 'eco', 'strong', 'social')
        imbalance: Maximum allowed imbalance (1-3, lower is stricter)
    
    Returns:
        partitions: Dictionary mapping partition ID to set of entity IDs
        entity_to_partition: Dictionary mapping entity ID to partition ID
        edge_cuts: Number of edge cuts
    """
    if len(triplets) == 0:
        print("Error: Cannot partition empty triplets")
        return {}, {}, 0
    
    print(f"\n{'='*60}")
    print(f"KaHIP Knowledge Graph Partitioning")
    print(f"{'='*60}")
    print(f"Mode: {mode}, Imbalance: {imbalance}%, Partitions: {num_parts}")
    
    # Check if KaHIP is installed
    if not check_kahip_installation():
        print("KaHIP not found. Falling back to random partitioning...")
        return partition_knowledge_graph_random(triplets, entity2id, relation2id, num_parts)
    
    # Create temporary directory for KaHIP files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Convert to METIS format
        print("Converting knowledge graph to METIS format...")
        metis_file = os.path.join(tmpdir, "graph.metis")
        num_vertices, num_edges, all_entities, entity_to_idx = convert_to_metis_format(
            triplets, entity2id, relation2id, metis_file
        )
        
        print(f"Graph stats: {num_vertices} vertices, {num_edges} edges")
        
        # Step 2: Run KaHIP
        print("Running KaHIP partitioning...")
        output_file = os.path.join(tmpdir, "partitions.txt")
        
        # Configure KaHIP parameters based on mode
        mode_params = {
            'fast': ['--fast'],
            'eco': ['--eco'],
            'strong': ['--strong'],
            'social': ['--social']
        }
        
        params = mode_params.get(mode, ['--strong'])
        
        # Add common parameters
        params.extend([
            str(metis_file),  # Input file
            '--k', str(num_parts),  # Number of partitions
            '--imbalance', str(imbalance),  # Imbalance parameter
            '--output_filename', output_file  # Output file
        ])
        
        try:
            # Try different KaHIP executable names
            executables = ['kaffpa', 'KaHIP/kaffpa', './kaffpa', 'kaffpaa']
            success = False
            result = None
            
            for executable in executables:
                try:
                    cmd = [executable] + params
                    print(f"Running: {' '.join(cmd)}")
                    
                    result = subprocess.run(cmd, 
                                          capture_output=True, 
                                          text=True, 
                                          timeout=600)  # 10 minute timeout
                    
                    if result.returncode == 0:
                        success = True
                        print("KaHIP completed successfully")
                        break
                    else:
                        print(f"Failed with {executable}: {result.stderr[:200]}")
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue
            
            if not success:
                print("All KaHIP executables failed. Falling back to random partitioning...")
                return partition_knowledge_graph_random(triplets, entity2id, relation2id, num_parts)
            
        except subprocess.TimeoutExpired:
            print("KaHIP timed out. Falling back to random partitioning...")
            return partition_knowledge_graph_random(triplets, entity2id, relation2id, num_parts)
        except Exception as e:
            print(f"KaHIP execution failed: {e}")
            print("Falling back to random partitioning...")
            return partition_knowledge_graph_random(triplets, entity2id, relation2id, num_parts)
        
        # Step 3: Parse KaHIP output
        if os.path.exists(output_file):
            print("Parsing KaHIP output...")
            partitions, entity_to_partition = parse_kahip_output(
                output_file, num_vertices, all_entities
            )
            
            if not partitions:
                print("Failed to parse KaHIP output. Falling back to random partitioning...")
                return partition_knowledge_graph_random(triplets, entity2id, relation2id, num_parts)
            
            # Count edge cuts
            edge_cuts = 0
            for h, r, t in triplets:
                if entity_to_partition.get(h) != entity_to_partition.get(t):
                    edge_cuts += 1
            
            print(f"KaHIP created {len(partitions)} partitions with {edge_cuts} edge cuts")
            
            # Extract quality info if available
            if result and result.stdout:
                import re
                cut_match = re.search(r'cut\s*:\s*(\d+)', result.stdout, re.IGNORECASE)
                if cut_match:
                    kahip_cut = int(cut_match.group(1))
                    print(f"  KaHIP reported cut: {kahip_cut}")
            
            return partitions, entity_to_partition, edge_cuts
        else:
            print("KaHIP output file not found. Falling back to random partitioning...")
            return partition_knowledge_graph_random(triplets, entity2id, relation2id, num_parts)


def save_partitions_kahip(partitions, triplets, entity2id, relation2id, edge_cuts, 
                          dataset_name, num_partitions, config_info=None):
    """
    Save KaHIP partitions to files in the project structure
    
    Args:
        partitions: Dictionary mapping partition ID to set of entity IDs
        triplets: Array of triplets
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        edge_cuts: Number of edge cuts
        dataset_name: Name of the dataset
        num_partitions: Number of partitions
        config_info: Additional configuration info
    """
    output_dir = f"data/partitions_kahip/{dataset_name}_{num_partitions}parts"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving partitions to {output_dir}...")
    
    # Clean up existing partition files
    for file in os.listdir(output_dir):
        if file.startswith("part-") or file == "dist_graph.json":
            os.remove(os.path.join(output_dir, file))
    
    all_nodes = set(entity2id.values())
    partition_list = list(partitions.values())
    
    # Create entity to partition mapping
    entity_to_partition = {}
    for part_id, nodes in enumerate(partition_list):
        for node in nodes:
            entity_to_partition[int(node)] = int(part_id)
    
    # Save partitions with internal edges only
    partition_stats = []
    for part_id, nodes in enumerate(partition_list):
        if not nodes:
            continue
            
        # Only include edges where both endpoints are in this partition
        part_triplets = [(h, r, t) for h, r, t in triplets if h in nodes and t in nodes]
        
        if not part_triplets:
            # Create empty partition with just nodes
            g = dgl.graph(([], []), num_nodes=len(nodes))
        else:
            src, rel, dst = zip(*part_triplets)
            g = dgl.graph((src, dst))
            g.edata['rel_type'] = torch.tensor(rel)
        
        num_nodes = len(nodes)
        num_relations = len(set(r for _, r, _ in part_triplets))
        num_edges = len(part_triplets)
        
        partition_stats.append({
            "part_id": part_id,
            "num_nodes": num_nodes,
            "num_relations": num_relations,
            "num_edges": num_edges
        })
        
        part_path = f"{output_dir}/part-{part_id}.bin"
        dgl.save_graphs(part_path, [g])
        
        # Save partition info as JSON
        part_info = {
            "part_id": part_id,
            "nodes": [int(node) for node in nodes],
            "triplets": [[int(h), int(r), int(t)] for h, r, t in part_triplets],
            "num_nodes": int(num_nodes),
            "num_relations": int(num_relations),
            "num_edges": int(num_edges)
        }
        with open(f"{output_dir}/part-{part_id}.json", "w") as f:
            json.dump(part_info, f, indent=2)
    
    # Calculate partition quality metrics
    print("\nCalculating partition quality metrics...")
    
    # Calculate balance metrics
    balance_metrics = PartitionQualityMetrics.calculate_balance_factor(partitions)
    
    edge_cut_ratio = edge_cuts / len(triplets) if len(triplets) > 0 else 0
    
    # Generate config file
    config = {
        "graph_name": f"kahip_{dataset_name}_{num_partitions}parts",
        "part_method": "KaHIP",
        "num_parts": len(partition_stats),
        "num_entities": int(len(all_nodes)),
        "num_relations": int(len(relation2id)),
        "edgecuts": int(edge_cuts),
        "kahip_config": {
            "mode": config_info.get('mode', 'strong') if config_info else 'strong',
            "imbalance": config_info.get('imbalance', 3) if config_info else 3
        },
        "partition_metrics": {
            "edge_cut_ratio": float(edge_cut_ratio),
            "balance_factor": float(balance_metrics['balance_factor']),
            "imbalance_ratio": float(balance_metrics['imbalance_ratio']),
            "partition_sizes": [int(s) for s in balance_metrics.get('partition_sizes', [])]
        },
        "partitions": {
            f"part-{stats['part_id']}": {
                "path": f"{output_dir}/part-{stats['part_id']}.bin",
                "num_entities": int(stats["num_nodes"]),
                "num_relations": int(stats["num_relations"]),
                "num_edges": int(stats["num_edges"])
            } 
            for stats in partition_stats
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"{output_dir}/dist_graph.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"KaHIP PARTITIONING COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Partitions created: {len(partition_stats)}")
    print(f"Edge cuts: {edge_cuts} ({edge_cut_ratio*100:.2f}%)")
    print(f"Balance factor: {balance_metrics['balance_factor']:.4f}")
    print(f"Mode: {config['kahip_config']['mode']}, Imbalance: {config['kahip_config']['imbalance']}")
    print(f"{'='*60}")
    
    return config


def main():
    """Main function for KaHIP partitioning"""
    parser = argparse.ArgumentParser(description="KaHIP Knowledge Graph Partitioning for ECML PKDD 2026")
    
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["FB15K", "FB15K-237", "YAGO3-10"],
                        help="Dataset name")
    parser.add_argument("--num-parts", type=int, required=True,
                        choices=[1, 4, 8, 12, 16],
                        help="Number of partitions")
    parser.add_argument("--mode", type=str, default="strong",
                        choices=["fast", "eco", "strong", "social"],
                        help="KaHIP partitioning mode")
    parser.add_argument("--imbalance", type=int, default=3,
                        choices=[1, 2, 3],
                        help="Maximum allowed imbalance (1-3, lower is stricter)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    
    # Get dataset config
    if args.dataset not in DATASET_CONFIGS:
        print(f"Error: Unknown dataset {args.dataset}")
        return
    
    dataset_config = DATASET_CONFIGS[args.dataset]
    data_dir = dataset_config["data_dir"]
    
    entities_dict = os.path.join(data_dir, "entities.dict")
    relations_dict = os.path.join(data_dir, "relations.dict")
    triplets_file = os.path.join(data_dir, "triplets.txt")
    
    print(f"\n{'='*60}")
    print(f"KaHIP Partitioning for ECML PKDD 2026")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of partitions: {args.num_parts}")
    print(f"Mode: {args.mode}")
    print(f"Imbalance: {args.imbalance}")
    print(f"Data directory: {data_dir}")
    print(f"{'='*60}\n")
    
    # Check if files exist
    if not os.path.exists(entities_dict):
        print(f"Error: Entities dictionary not found: {entities_dict}")
        print("Please ensure the dataset is properly set up.")
        return
    
    if not os.path.exists(relations_dict):
        print(f"Error: Relations dictionary not found: {relations_dict}")
        return
    
    if not os.path.exists(triplets_file):
        print(f"Error: Triplets file not found: {triplets_file}")
        return
    
    # Load dictionaries
    print("Loading dictionaries...")
    entity2id = load_dict(entities_dict)
    relation2id = load_dict(relations_dict)
    
    if not entity2id:
        print("Error: Failed to load entity dictionary")
        return
    
    if not relation2id:
        print("Error: Failed to load relation dictionary")
        return
    
    print(f"Loaded {len(entity2id)} entities and {len(relation2id)} relations")
    
    # Read triplets
    print("Loading triplets...")
    triplets = read_triplets(triplets_file, entity2id, relation2id)
    
    if len(triplets) == 0:
        print("Error: No valid triplets found!")
        return
    
    print(f"Loaded {len(triplets)} triplets")
    
    # Check if we have enough data for the requested partitions
    unique_entities = len(set([h for h, _, _ in triplets] + [t for _, _, t in triplets]))
    if unique_entities < args.num_parts:
        print(f"Warning: Only {unique_entities} unique entities, "
              f"reducing partitions to {unique_entities}")
        args.num_parts = unique_entities
    
    # Run KaHIP partitioning
    partitions, entity_to_partition, edge_cuts = partition_knowledge_graph_kahip(
        triplets, entity2id, relation2id, args.num_parts, 
        mode=args.mode, imbalance=args.imbalance
    )
    
    if partitions:
        # Save partitions
        config_info = {
            'mode': args.mode,
            'imbalance': args.imbalance,
            'algorithm': 'KaHIP'
        }
        
        save_partitions_kahip(
            partitions, triplets, entity2id, relation2id, 
            edge_cuts, args.dataset, args.num_parts, config_info
        )
    else:
        print("KaHIP partitioning failed - no partitions were created.")


if __name__ == "__main__":
    main()