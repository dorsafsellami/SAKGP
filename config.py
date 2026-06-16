# config.py - Unified Configuration for Distributed KGE Training
import os
import torch

# =============================================================================
# MODEL-SPECIFIC CONFIGURATIONS
# =============================================================================

TRANSE_CONFIG = {
    "model_name": "transe",
    "num_entities": 14951,  # Will be updated per dataset
    "num_relations": 1345,
    "embedding_dim": 128,
    "margin": 1.0,
    "norm": 1,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "batch_size": 2048,
    "max_epochs": 200,
    "loss_type": "margin_ranking",
}

RGCN_CONFIG = {
    "model_name": "rgcn",
    "num_entities": 14951,
    "num_relations": 1345,
    "embedding_dim": 128,
    "num_bases": 4,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 1024,
    "max_epochs": 200,
    "scoring_function": "distmult",
}

KGTRANSFORMER_CONFIG = {
    "model_name": "kgtransformer",
    "num_entities": 14951,
    "num_relations": 1345,
    "embedding_dim": 64,
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "dropout": 0.2,
    "learning_rate": 0.0005,
    "batch_size": 512,
    "max_epochs": 200,
    "use_relation_attention": True,
}

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASET_CONFIGS = {
    "FB15K": {
        "num_entities": 14951,
        "num_relations": 1345,
        "triplet_count": 592213,
        "data_dir": "data/FB15K",
    },
    "FB15K-237": {
        "num_entities": 14541,
        "num_relations": 237,
        "triplet_count": 310079,
        "data_dir": "data/FB15K-237",
    },
    "YAGO3-10": {
        "num_entities": 123182,
        "num_relations": 37,
        "triplet_count": 1179040,
        "data_dir": "data/YAGO3-10",
    }
}

# =============================================================================
# DISTRIBUTED TRAINING CONFIGURATION
# =============================================================================

# Grid'5000 node configuration
NODE_CONFIG = {
    "graffiti-1": {"ip": "graffiti-1.nancy.grid5000.fr", "gpus": 4},
    "graffiti-2": {"ip": "graffiti-2.nancy.grid5000.fr", "gpus": 4},
    "graffiti-3": {"ip": "graffiti-3.nancy.grid5000.fr", "gpus": 4},
    "graffiti-4": {"ip": "graffiti-4.nancy.grid5000.fr", "gpus": 4},
}

def get_node_allocation(num_workers):
    """
    Allocate workers across nodes based on number of workers
    Returns list of (node_ip, gpu_id, rank) tuples
    """
    if num_workers == 1:
        return [("graffiti-1.nancy.grid5000.fr", 0, 0)]
    
    elif num_workers == 4:
        return [
            ("graffiti-1.nancy.grid5000.fr", 0, 0),
            ("graffiti-1.nancy.grid5000.fr", 1, 1),
            ("graffiti-1.nancy.grid5000.fr", 2, 2),
            ("graffiti-1.nancy.grid5000.fr", 3, 3),
        ]
    
    elif num_workers == 8:
        return [
            ("graffiti-1.nancy.grid5000.fr", 0, 0),
            ("graffiti-1.nancy.grid5000.fr", 1, 1),
            ("graffiti-1.nancy.grid5000.fr", 2, 2),
            ("graffiti-1.nancy.grid5000.fr", 3, 3),
            ("graffiti-2.nancy.grid5000.fr", 0, 4),
            ("graffiti-2.nancy.grid5000.fr", 1, 5),
            ("graffiti-2.nancy.grid5000.fr", 2, 6),
            ("graffiti-2.nancy.grid5000.fr", 3, 7),
        ]
    
    elif num_workers == 12:
        return [
            ("graffiti-1.nancy.grid5000.fr", 0, 0),
            ("graffiti-1.nancy.grid5000.fr", 1, 1),
            ("graffiti-1.nancy.grid5000.fr", 2, 2),
            ("graffiti-1.nancy.grid5000.fr", 3, 3),
            ("graffiti-2.nancy.grid5000.fr", 0, 4),
            ("graffiti-2.nancy.grid5000.fr", 1, 5),
            ("graffiti-2.nancy.grid5000.fr", 2, 6),
            ("graffiti-2.nancy.grid5000.fr", 3, 7),
            ("graffiti-3.nancy.grid5000.fr", 0, 8),
            ("graffiti-3.nancy.grid5000.fr", 1, 9),
            ("graffiti-3.nancy.grid5000.fr", 2, 10),
            ("graffiti-3.nancy.grid5000.fr", 3, 11),
        ]
    
    elif num_workers == 16:
        return [
            ("graffiti-1.nancy.grid5000.fr", 0, 0),
            ("graffiti-1.nancy.grid5000.fr", 1, 1),
            ("graffiti-1.nancy.grid5000.fr", 2, 2),
            ("graffiti-1.nancy.grid5000.fr", 3, 3),
            ("graffiti-2.nancy.grid5000.fr", 0, 4),
            ("graffiti-2.nancy.grid5000.fr", 1, 5),
            ("graffiti-2.nancy.grid5000.fr", 2, 6),
            ("graffiti-2.nancy.grid5000.fr", 3, 7),
            ("graffiti-3.nancy.grid5000.fr", 0, 8),
            ("graffiti-3.nancy.grid5000.fr", 1, 9),
            ("graffiti-3.nancy.grid5000.fr", 2, 10),
            ("graffiti-3.nancy.grid5000.fr", 3, 11),
            ("graffiti-4.nancy.grid5000.fr", 0, 12),
            ("graffiti-4.nancy.grid5000.fr", 1, 13),
            ("graffiti-4.nancy.grid5000.fr", 2, 14),
            ("graffiti-4.nancy.grid5000.fr", 3, 15),
        ]
    
    else:
        raise ValueError(f"Unsupported number of workers: {num_workers}")

# =============================================================================
# SYNCHRONIZATION STRATEGIES (BSP and ASP removed)
# =============================================================================

SYNC_CONFIGS = {
    "ssp": {
        "communication_strategy": "ssp",
        "ssp_staleness": 2,
        "sync_interval": 5,
        "description": "Stale Synchronous Parallel - allows bounded staleness"
    }
}

# Default synchronization strategy
DEFAULT_SYNC_CONFIG = SYNC_CONFIGS["ssp"]

# =============================================================================
# EXPERIMENT TRACKING
# =============================================================================

EXPERIMENT_REGISTRY = {
    "completed": [],
    "results_dir": "results/",
    "logs_dir": "logs/",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_config(model_name, dataset_name):
    """Get configuration for specific model and dataset"""
    if model_name == "transe":
        config = TRANSE_CONFIG.copy()
    elif model_name == "rgcn":
        config = RGCN_CONFIG.copy()
    elif model_name == "kgtransformer":
        config = KGTRANSFORMER_CONFIG.copy()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Update with dataset info
    dataset_config = DATASET_CONFIGS[dataset_name]
    config["num_entities"] = dataset_config["num_entities"]
    config["num_relations"] = dataset_config["num_relations"]
    config["data_dir"] = dataset_config["data_dir"]
    
    return config

def get_partition_dir(partition_method, dataset_name, num_partitions):
    """Get partition directory for specific configuration"""
    method_map = {
        "metis": "partitions_metis",
        "kahip": "partitions_kahip", 
        "sakgp": "partitions"
    }
    return f"data/{method_map[partition_method]}"

def enable_performance_optimizations():
    """Enable all performance optimizations"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    
    # Enable flash attention if available
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

enable_performance_optimizations()

print("✓ Unified configuration loaded for experiments")
print(f"  - Models: TransE, RGCN, KGTransformer")
print(f"  - Datasets: FB15K, FB15K-237, YAGO3-10")
print(f"  - Partitioning: METIS, KaHIP, SAKGP")
print(f"  - Worker counts: 1, 4, 8, 12, 16")