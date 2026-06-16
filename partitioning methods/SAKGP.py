import os
import torch
import networkx as nx
import numpy as np
from collections import defaultdict
import dgl
import json
import random
import warnings
import time
from config import DATASET_CONFIGS

warnings.filterwarnings('ignore')

def load_dict(file_path):
    """Load dictionary file with format: id\tvalue"""
    loaded_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    uid, val = parts[0], parts[1]
                    loaded_dict[val] = int(uid)
    return loaded_dict

def read_triplets(file_path, entity2id, relation2id):
    """Read triplets file and convert to numeric IDs"""
    triplets = []
    line_count = 0
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
                        print(f"Skipping invalid triplet on line {line_count}: {head} {relation} {tail}")
                else:
                    print(f"Skipping malformed line {line_count}: {line}")
    print(f"Processed {line_count} lines, found {len(triplets)} valid triplets")
    return np.array(triplets)

def check_triplets_file_format(file_path):
    """Check the format of the triplets file"""
    print(f"Checking triplets file format: {file_path}")
    
    if not os.path.exists(file_path):
        print("Triplets file does not exist")
        return False
    
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        if not first_line:
            print("Triplets file is empty")
            return False
        
        parts = first_line.split()
        print(f"First line: {first_line}")
        print(f"Parts: {parts}")
        
        if len(parts) < 3:
            print("Triplets file has incorrect format (expected: entity relation entity)")
            return False
    
    return True

def generate_triplets_file(entity2id, relation2id, output_file, num_triplets):
    """Generate sample triplets file for testing"""
    print(f"Generating sample triplets file with {num_triplets} triplets...")
    
    entities = list(entity2id.keys())
    relations = list(relation2id.keys())
    
    if len(entities) < 2 or len(relations) < 1:
        print("Not enough entities or relations to generate triplets")
        return False
    
    try:
        with open(output_file, 'w') as f:
            for i in range(num_triplets):
                head = random.choice(entities)
                tail = random.choice(entities)
                relation = random.choice(relations)
                f.write(f"{head}\t{relation}\t{tail}\n")
        print(f"Generated {num_triplets} sample triplets in {output_file}")
        return True
    except Exception as e:
        print(f"Error generating triplets file: {e}")
        return False

def prepare_relation_features(G, triplets):
    """Pre-compute features for all relations"""
    entity_set = set(G.nodes())
    relation_to_triplets = defaultdict(list)
    
    for h, r, t in triplets:
        relation_to_triplets[r].append((h, t))
    
    relation_features = {}
    
    for r, edges in relation_to_triplets.items():
        if len(edges) == 0:
            continue
            
        # Feature 1: RCS (Relation Connectivity Score)
        degrees_sum = sum(G.degree(h) + G.degree(t) for h, t in edges)
        rcs = degrees_sum / len(edges)
        
        # Feature 2: RSS (Relation Semantic Specificity)
        entity_counter = defaultdict(int)
        for h, t in edges:
            entity_counter[h] += 1
            entity_counter[t] += 1
        
        total = sum(entity_counter.values())
        if total > 0:
            entropy = -sum((count/total) * np.log2(count/total + 1e-10) 
                          for count in entity_counter.values() if count > 0)
            rss = 1 - entropy / np.log2(len(entity_set)) if len(entity_set) > 1 else 1
        else:
            rss = 1
        
        # Feature 3: RSS' (Relation Structural Similarity)
        closed_triangles = 0
        for h, t in edges:
            h_successors = set(G.successors(h))
            t_successors = set(G.successors(t))
            common_neighbors = h_successors.intersection(t_successors)
            closed_triangles += len(common_neighbors)
        
        rss_prime = closed_triangles / len(edges) if len(edges) > 0 else 0
        
        relation_features[r] = {
            'rcs': rcs,
            'rss': rss,
            'rss_prime': rss_prime,
            'num_edges': len(edges)
        }
    
    return relation_features

def calculate_weights_from_features(relation_features, alpha, beta, gamma):
    """Calculate relation weights from features using given parameters"""
    # Extract all feature values for normalization
    all_rcs = [feat['rcs'] for feat in relation_features.values()]
    all_rss = [feat['rss'] for feat in relation_features.values()]
    all_rss_prime = [feat['rss_prime'] for feat in relation_features.values()]
    
    # Calculate ranges for normalization
    rcs_min, rcs_max = min(all_rcs), max(all_rcs)
    rss_min, rss_max = min(all_rss), max(all_rss)
    rss_prime_min, rss_prime_max = min(all_rss_prime), max(all_rss_prime)
    
    rcs_range = rcs_max - rcs_min if rcs_max > rcs_min else 1
    rss_range = rss_max - rss_min if rss_max > rss_min else 1
    rss_prime_range = rss_prime_max - rss_prime_min if rss_prime_max > rss_prime_min else 1
    
    relation_weights = {}
    
    for r, features in relation_features.items():
        # Normalize features
        rcs_norm = (features['rcs'] - rcs_min) / rcs_range
        rss_norm = (features['rss'] - rss_min) / rss_range
        rss_prime_norm = (features['rss_prime'] - rss_prime_min) / rss_prime_range
        
        # Calculate weight
        weight = alpha * rcs_norm + beta * rss_norm + gamma * rss_prime_norm
        
        # Ensure weight is positive and not too small
        weight = max(0.01, weight)
        relation_weights[r] = weight
    
    return relation_weights

def estimate_partition_quality(G, triplets, relation_weights, num_partitions=8, sample_size=2000):
    """Estimate partition quality using a simplified clustering approach"""
    if len(triplets) == 0:
        return float('inf')
    
    # Sample a subset for faster estimation
    if len(triplets) > sample_size:
        sampled_indices = np.random.choice(len(triplets), sample_size, replace=False)
        sample_triplets = triplets[sampled_indices]
    else:
        sample_triplets = triplets
    
    # Create node weights based on incident relation weights
    node_weights = defaultdict(float)
    for h, r, t in sample_triplets:
        weight = relation_weights.get(r, 0.5)
        node_weights[h] += weight
        node_weights[t] += weight
    
    # Select seeds based on node weights
    sorted_nodes = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)
    seeds = [node for node, _ in sorted_nodes[:num_partitions]]
    
    if len(seeds) < num_partitions:
        return float('inf')
    
    # Simple label propagation simulation
    partitions = {seed: {seed} for seed in seeds}
    node_to_partition = {seed: seed for seed in seeds}
    
    # Assign remaining nodes to nearest seed based on weighted connections
    remaining_nodes = set(G.nodes()) - set(seeds)
    
    for node in remaining_nodes:
        best_partition = None
        best_score = -float('inf')
        
        for seed in seeds:
            # Calculate connection strength to this partition
            connection_strength = 0
            for neighbor in G.neighbors(node):
                if neighbor in partitions[seed]:
                    edge_data = G.get_edge_data(node, neighbor)
                    if edge_data:
                        for edge_info in edge_data.values():
                            rel = edge_info.get('relation', 0)
                            connection_strength += relation_weights.get(rel, 0.5)
            
            # Penalize large partitions
            partition_size = len(partitions[seed])
            size_penalty = 0.05 * partition_size
            
            score = connection_strength - size_penalty
            
            if score > best_score:
                best_score = score
                best_partition = seed
        
        if best_partition:
            partitions[best_partition].add(node)
            node_to_partition[node] = best_partition
    
    # Estimate edge cuts
    edge_cuts = 0
    for h, r, t in sample_triplets:
        if node_to_partition.get(h) != node_to_partition.get(t):
            edge_cuts += 1
    
    # Calculate balance score
    partition_sizes = [len(nodes) for nodes in partitions.values()]
    if partition_sizes:
        balance_score = max(partition_sizes) / (sum(partition_sizes) / len(partition_sizes))
    else:
        balance_score = float('inf')
    
    # Combined quality score (lower is better)
    quality_score = edge_cuts * balance_score
    
    return quality_score

def optimize_parameters_for_partitioning(G, triplets, relation_features, num_partitions=8):
    """Optimize α, β, γ parameters using differential evolution"""
    
    def objective_function(params):
        """Objective function to minimize (edge cuts + imbalance)"""
        alpha, beta, gamma = params
        
        # Ensure parameters sum to 1
        total = alpha + beta + gamma
        if total == 0:
            return float('inf')
        
        alpha, beta, gamma = alpha/total, beta/total, gamma/total
        
        # Calculate relation weights
        relation_weights = calculate_weights_from_features(relation_features, alpha, beta, gamma)
        
        # Estimate partition quality
        quality_score = estimate_partition_quality(G, triplets, relation_weights, num_partitions)
        
        return quality_score
    
    # Define bounds for parameters (0 to 1)
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    
    # Use differential evolution for global optimization
    try:
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=50,
            popsize=10,
            seed=42,
            disp=False
        )
        
        # Normalize parameters
        alpha, beta, gamma = result.x
        total = alpha + beta + gamma
        if total == 0:
            alpha, beta, gamma = 0.33, 0.33, 0.34
        else:
            alpha, beta, gamma = alpha/total, beta/total, gamma/total
        
        return alpha, beta, gamma, result.fun
    
    except Exception as e:
        print(f"Optimization failed: {e}. Using default parameters.")
        return 0.33, 0.33, 0.34, float('inf')

def compute_relation_weights_with_optimization(G, triplets, optimization_level='auto'):
    """Compute relation weights with automatic parameter optimization"""
    print("Computing relation features...")
    relation_features = prepare_relation_features(G, triplets)
    
    if len(relation_features) < 2:
        print("Warning: Not enough relations for optimization. Using default parameters.")
        relation_weights = {r: 1.0 for r in relation_features.keys()}
        return relation_weights, (0.33, 0.33, 0.34)
    
    # Determine optimization level
    if optimization_level == 'fast':
        max_relations_for_opt = 50
        num_iterations = 20
    elif optimization_level == 'balanced':
        max_relations_for_opt = 150
        num_iterations = 75
    else:  # 'auto' or 'full'
        max_relations_for_opt = min(200, len(relation_features))
        num_iterations = 100
    
    print(f"Optimizing parameters for {len(relation_features)} relations...")
    
    # For very large graphs, use a subset for optimization
    if len(relation_features) > max_relations_for_opt:
        print(f"Using subset of {max_relations_for_opt} relations for optimization...")
        # Select most representative relations
        relation_edges = [(r, feat['num_edges']) for r, feat in relation_features.items()]
        relation_edges.sort(key=lambda x: x[1], reverse=True)
        selected_relations = [r for r, _ in relation_edges[:max_relations_for_opt]]
        
        # Create subset of features
        subset_features = {r: relation_features[r] for r in selected_relations}
        
        # Create subset of triplets for estimation
        subset_triplet_indices = []
        for i, (h, r, t) in enumerate(triplets):
            if r in selected_relations:
                subset_triplet_indices.append(i)
        
        if len(subset_triplet_indices) > 0:
            subset_triplets = triplets[subset_triplet_indices]
            
            # Optimize on subset
            alpha, beta, gamma, score = optimize_parameters_for_partitioning(
                G, subset_triplets, subset_features, num_partitions=8
            )
        else:
            alpha, beta, gamma = 0.33, 0.33, 0.34
    else:
        # Optimize on all relations
        alpha, beta, gamma, score = optimize_parameters_for_partitioning(
            G, triplets, relation_features, num_partitions=8
        )
    
    print(f"Optimized parameters: α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}")
    print(f"Estimated quality score: {score:.4f}")
    
    # Calculate final weights with optimized parameters
    relation_weights = calculate_weights_from_features(relation_features, alpha, beta, gamma)
    
    # Analyze weight distribution
    weight_values = list(relation_weights.values())
    print(f"Relation weight distribution:")
    print(f"  Min: {min(weight_values):.4f}")
    print(f"  Max: {max(weight_values):.4f}")
    print(f"  Mean: {np.mean(weight_values):.4f}")
    print(f"  Std: {np.std(weight_values):.4f}")
    
    # Print top relations by weight
    sorted_relations = sorted(relation_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 relations by weight:")
    for r, weight in sorted_relations:
        feat = relation_features[r]
        print(f"  Relation {r}: weight={weight:.4f}, edges={feat['num_edges']}, "
              f"RCS={feat['rcs']:.2f}, RSS={feat['rss']:.2f}, RSS'={feat['rss_prime']:.2f}")
    
    return relation_weights, (alpha, beta, gamma)


class PartitionQualityMetrics:
    """Metrics for evaluating partition quality"""
    
    @staticmethod
    def calculate_edge_cut_ratio(partitions: dict, triplets: np.ndarray, 
                                 entity_to_partition: dict) -> float:
        """
        Calculate edge cut ratio - percentage of edges crossing partitions
        
        Args:
            partitions: Dictionary mapping partition ID to set of entity IDs
            triplets: Array of triplets (head, relation, tail)
            entity_to_partition: Dictionary mapping entity ID to partition ID
        
        Returns:
            Edge cut ratio (0.0 to 1.0)
        """
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
    def calculate_balance_factor(partitions: dict) -> dict:
        """
        Calculate balance factor metrics
        
        Args:
            partitions: Dictionary mapping partition ID to set of entity IDs
        
        Returns:
            Dictionary with balance metrics
        """
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
        
        # Balance factor (ideal = 1.0)
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
    
    @staticmethod
    def calculate_relation_distribution(partitions: dict, triplets: np.ndarray,
                                        relation2id: dict, 
                                        entity_to_partition: dict) -> dict:
        """
        Calculate relation distribution across partitions
        
        Args:
            partitions: Dictionary mapping partition ID to set of entity IDs
            triplets: Array of triplets (head, relation, tail)
            relation2id: Dictionary mapping relation string to ID
            entity_to_partition: Dictionary mapping entity ID to partition ID
        
        Returns:
            Dictionary with relation distribution metrics
        """
        if len(triplets) == 0:
            return {'relation_balance': 1.0, 'partition_coverage': {}}
        
        # Count relations per partition
        partition_relation_counts = defaultdict(lambda: defaultdict(int))
        
        for h, r, t in triplets:
            h_part = entity_to_partition.get(h)
            t_part = entity_to_partition.get(t)
            
            # Count relation in each partition it appears in
            if h_part is not None:
                partition_relation_counts[h_part][r] += 1
            if t_part is not None and t_part != h_part:
                partition_relation_counts[t_part][r] += 1
        
        # Calculate relation balance score
        relation_balance_scores = []
        total_relations = len(relation2id)
        
        for r in relation2id.values():
            r_id = r
            # Count how many partitions this relation appears in
            partitions_with_relation = 0
            total_relation_count = 0
            
            for part_id in partitions:
                count = partition_relation_counts[part_id].get(r_id, 0)
                if count > 0:
                    partitions_with_relation += 1
                    total_relation_count += count
            
            if partitions_with_relation > 0:
                # Ideal: relation appears in all partitions equally
                ideal_per_partition = total_relation_count / len(partitions)
                actual_distribution = []
                
                for part_id in partitions:
                    count = partition_relation_counts[part_id].get(r_id, 0)
                    actual_distribution.append(count)
                
                # Calculate distribution evenness (1.0 = perfectly even)
                if ideal_per_partition > 0:
                    distribution_diff = sum(abs(c - ideal_per_partition) for c in actual_distribution)
                    max_diff = 2 * ideal_per_partition * (len(partitions) - 1)
                    balance_score = 1.0 - (distribution_diff / max_diff if max_diff > 0 else 0)
                    relation_balance_scores.append(balance_score)
        
        overall_balance = np.mean(relation_balance_scores) if relation_balance_scores else 1.0
        
        # Calculate partition coverage
        partition_coverage = {}
        for part_id in partitions:
            unique_relations = len(partition_relation_counts[part_id])
            coverage = unique_relations / total_relations if total_relations > 0 else 0
            partition_coverage[part_id] = {
                'unique_relations': unique_relations,
                'coverage': coverage,
                'total_edges': sum(partition_relation_counts[part_id].values())
            }
        
        return {
            'relation_balance': float(overall_balance),
            'partition_coverage': partition_coverage,
            'total_unique_relations': total_relations
        }


def partition_knowledge_graph(triplets, entity2id, relation2id, num_parts=num_part, optimization_level='auto'):
    """Partition the knowledge graph using SALP algorithm with optimized parameters"""
    if len(triplets) == 0:
        print("Error: Cannot partition empty triplets")
        return {}, {}, {}, (0.33, 0.33, 0.34)
        
    # Step 1: Detect symmetric relations
    def detect_symmetric_relations(triplets):
        relation_pairs = defaultdict(set)
        for h, r, t in triplets:
            relation_pairs[r].add((h, t))

        symmetric_relations = set()
        for r, pairs in relation_pairs.items():
            for (h, t) in pairs:
                if (t, h) in pairs:
                    symmetric_relations.add(r)
                    break
        return symmetric_relations

    symmetric_relations = detect_symmetric_relations(triplets)
    print(f"Detected symmetric relations: {symmetric_relations}")

    # Step 2: Preprocess for degree calculation
    def preprocess_for_degree(triplets, symmetric_relations):
        unique_edges = set()
        for head, relation, tail in triplets:
            if relation in symmetric_relations:
                small, large = min(head, tail), max(head, tail)
                unique_edges.add((small, relation, large))
            else:
                unique_edges.add((head, relation, tail))

        in_degree_count = defaultdict(int)
        for head, relation, tail in unique_edges:
            in_degree_count[tail] += 1

        return in_degree_count

    in_degree_count = preprocess_for_degree(triplets, symmetric_relations)

    # Step 3: Select top-N seeds
    N_clusters = num_parts
    sorted_nodes = sorted(in_degree_count.items(), key=lambda x: x[1], reverse=True)
    
    # Get all nodes that have edges
    valid_seeds = [node for node, degree in sorted_nodes if degree > 0]
    
    # Adjust number of partitions if we don't have enough nodes with edges
    actual_num_parts = min(num_parts, len(valid_seeds))
    if actual_num_parts < num_parts:
        print(f"Warning: Only {len(valid_seeds)} nodes have edges. Reducing partitions from {num_parts} to {actual_num_parts}")
        num_parts = actual_num_parts
    
    if num_parts == 0:
        print("Error: No nodes with edges found for partitioning!")
        return {}, {}, {}, (0.33, 0.33, 0.34)
    
    seeds = valid_seeds[:num_parts]
    print(f"Selected {len(seeds)} seeds: {seeds}")

    # Step 4: Construct full KG
    G = nx.MultiDiGraph()
    for head, relation, tail in triplets:
        G.add_edge(head, tail, relation=relation)

    print(f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Step 5: Compute relation weights with optimization
    print("Computing relation weights with automatic parameter optimization...")
    relation_weights, (alpha, beta, gamma) = compute_relation_weights_with_optimization(
        G, triplets, optimization_level
    )
    print(f"Relation weights computed for {len(relation_weights)} relations")

    # Step 6: Initial partitioning
    partitions = {seed: {seed} for seed in seeds}
    node_to_partition = {seed: seed for seed in seeds}

    all_nodes = set(G.nodes())
    for node in all_nodes:
        if node not in node_to_partition:
            # Assign to partition with strongest connection
            best_partition = None
            best_connection = -1
            
            for seed in seeds:
                connection_strength = 0
                for neighbor in G.neighbors(node):
                    if neighbor in partitions[seed]:
                        edge_data = G.get_edge_data(node, neighbor)
                        if edge_data:
                            for edge_info in edge_data.values():
                                rel = edge_info.get('relation', 0)
                                connection_strength += relation_weights.get(rel, 0.5)
                
                if connection_strength > best_connection:
                    best_connection = connection_strength
                    best_partition = seed
            
            if best_partition is None:
                best_partition = random.choice(seeds)
            
            partitions[best_partition].add(node)
            node_to_partition[node] = best_partition

    # Step 7: SVP Score Calculation
    def compute_svp_score(node, node_partition, lambda_size=0.05):
        local_weight = 0
        remote_weight = 0
        
        if node not in G:
            return 0
            
        for neighbor in G.neighbors(node):
            edge_data_list = G.get_edge_data(node, neighbor)
            if edge_data_list:
                for edge_data in edge_data_list.values():
                    weight = relation_weights.get(edge_data.get('relation', 0), 0.5)
                    if node_to_partition.get(neighbor) == node_partition:
                        local_weight += weight
                    else:
                        remote_weight += weight
        
        partition_size = len(partitions.get(node_partition, set()))
        partition_size_penalty = lambda_size * partition_size
        return local_weight - remote_weight - partition_size_penalty

    # Step 8: Semantic-Aware Label Propagation
    def salp(max_iter=100, theta=0.0):
        for iteration in range(max_iter):
            changes = 0
            nodes = list(G.nodes())
            random.shuffle(nodes)
            
            for node in nodes:
                current_partition = node_to_partition.get(node)
                if current_partition is None:
                    continue
                    
                current_score = compute_svp_score(node, current_partition)
                
                if current_score <= theta:
                    best_partition = current_partition
                    best_score = current_score
                    
                    for candidate_partition in seeds:
                        if candidate_partition == current_partition:
                            continue
                        score = compute_svp_score(node, candidate_partition)
                        if score > best_score:
                            best_partition = candidate_partition
                            best_score = score
                    
                    if best_partition != current_partition:
                        if current_partition in partitions and node in partitions[current_partition]:
                            partitions[current_partition].remove(node)
                        if best_partition not in partitions:
                            partitions[best_partition] = set()
                        partitions[best_partition].add(node)
                        node_to_partition[node] = best_partition
                        changes += 1
            
            print(f"Iteration {iteration + 1}: {changes} changes")
            if changes == 0:
                break

    # Run SALP
    print("Starting SALP partitioning...")
    salp()
    
    # Remove empty partitions
    partitions = {k: v for k, v in partitions.items() if len(v) > 0}
    
    # Calculate partition quality metrics
    print("\n=== CALCULATING PARTITION QUALITY METRICS ===")
    
    # Create entity to partition mapping
    entity_to_partition = {}
    for part_id, nodes in partitions.items():
        for node in nodes:
            entity_to_partition[node] = part_id
    
    # Calculate edge cut ratio
    edge_cut_ratio = PartitionQualityMetrics.calculate_edge_cut_ratio(
        partitions, triplets, entity_to_partition
    )
    
    # Calculate balance factor
    balance_metrics = PartitionQualityMetrics.calculate_balance_factor(partitions)
    
    # Calculate relation distribution
    relation_metrics = PartitionQualityMetrics.calculate_relation_distribution(
        partitions, triplets, relation2id, entity_to_partition
    )
    
    # Print metrics
    print(f"Edge Cut Ratio: {edge_cut_ratio:.4f} ({edge_cut_ratio*100:.1f}%)")
    print(f"Balance Factor: {balance_metrics['balance_factor']:.4f} (Ideal: 1.0)")
    print(f"Imbalance Ratio: {balance_metrics['imbalance_ratio']:.4f} (Ideal: 1.0)")
    print(f"Relation Balance: {relation_metrics['relation_balance']:.4f} (Ideal: 1.0)")
    print(f"Partition Sizes: Max={balance_metrics['max_size']}, Min={balance_metrics['min_size']}, "
          f"Avg={balance_metrics['avg_size']:.1f}, Std={balance_metrics['std_dev']:.1f}")
    
    # Print partition coverage
    print("\nPartition Coverage:")
    for part_id, coverage in relation_metrics['partition_coverage'].items():
        print(f"  Partition {part_id}: {coverage['unique_relations']} relations "
              f"({coverage['coverage']*100:.1f}% coverage), {coverage['total_edges']} edges")
    
    return partitions, node_to_partition, relation_weights, (alpha, beta, gamma)


def save_partitions_sakgp(partitions, triplets, entity2id, relation2id, edge_cuts, 
                          optimized_params, dataset_name, num_partitions):
    """
    Save SAKGP partitions to files in the project structure
    
    Args:
        partitions: Dictionary mapping partition ID to set of entity IDs
        triplets: Array of triplets
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        edge_cuts: Number of edge cuts
        optimized_params: Optimized parameters (alpha, beta, gamma)
        dataset_name: Name of the dataset
        num_partitions: Number of partitions
    """
    output_dir = f"data/partitions_sakgp/{dataset_name}_{num_partitions}parts"
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
        "graph_name": f"sakgp_{dataset_name}_{num_partitions}parts",
        "part_method": "SAKGP",
        "num_parts": len(partition_stats),
        "num_entities": int(len(all_nodes)),
        "num_relations": int(len(relation2id)),
        "edgecuts": int(edge_cuts),
        "optimized_parameters": {
            "alpha": float(optimized_params[0]),
            "beta": float(optimized_params[1]),
            "gamma": float(optimized_params[2])
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
    print(f"SAKGP PARTITIONING COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Partitions created: {len(partition_stats)}")
    print(f"Edge cuts: {edge_cuts} ({edge_cut_ratio*100:.2f}%)")
    print(f"Balance factor: {balance_metrics['balance_factor']:.4f}")
    print(f"Optimized parameters: α={optimized_params[0]:.4f}, "
          f"β={optimized_params[1]:.4f}, γ={optimized_params[2]:.4f}")
    print(f"{'='*60}")
    
    return config



def main():
    """Main function for SAKGP partitioning"""
    parser = argparse.ArgumentParser(description="SAKGP Knowledge Graph Partitioning for ECML PKDD 2026")
    
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["FB15K", "FB15K-237", "YAGO3-10"],
                        help="Dataset name")
    parser.add_argument("--num-parts", type=int, required=True,
                        choices=[1, 4, 8, 12, 16],
                        help="Number of partitions")
    parser.add_argument("--optimize", action="store_true", default=True,
                        help="Optimize relation weights (default: True)")
    parser.add_argument("--no-optimize", action="store_false", dest="optimize",
                        help="Disable relation weight optimization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
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
    print(f"SAKGP Partitioning for ECML PKDD 2026")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of partitions: {args.num_parts}")
    print(f"Optimization: {'Enabled' if args.optimize else 'Disabled'}")
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
    
    # Run SAKGP partitioning
    partitions, node_to_partition, edge_cuts, optimized_params = partition_knowledge_graph_sakgp(
        triplets, entity2id, relation2id, args.num_parts, optimize=args.optimize
    )
    
    if partitions:
        # Save partitions
        save_partitions_sakgp(
            partitions, triplets, entity2id, relation2id, 
            edge_cuts, optimized_params, args.dataset, args.num_parts
        )
    else:
        print("SAKGP partitioning failed - no partitions were created.")


if __name__ == "__main__":
    main()