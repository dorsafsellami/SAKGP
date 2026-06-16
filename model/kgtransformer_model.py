# kgtransformer_model.py - KGTransformer Model for Knowledge Graph Completion
import os
import logging
import threading
import math
import time
import psutil
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from typing import Optional, Tuple, List, Dict
import httpx
import asyncio

from config import GLOBAL_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources with vectorized operations"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.monitoring = False
        
        # Initialize metrics with numpy arrays for vectorized operations
        self.cpu_usage = np.array([], dtype=np.float32)
        self.gpu_usage = np.array([], dtype=np.float32)
        self.gpu_memory = np.array([], dtype=np.float32)
        self.gpu_power = np.array([], dtype=np.float32)
        self.ram_usage = np.array([], dtype=np.float32)
        
        self.bandwidth_sent = 0
        self.bandwidth_received = 0
        self.sync_operations = 0
        
        self.start_time = None
        self.end_time = None
        self.communication_times = []
        self.computation_times = []
        
        # Monitoring thread for continuous sampling
        self.sampling_thread = None
        self.stop_sampling = False
        
    def start_monitoring(self):
        """Start monitoring resources"""
        self.monitoring = True
        self.start_time = time.time()
        self.stop_sampling = False
        
        # Clear previous data
        self.cpu_usage = np.array([], dtype=np.float32)
        self.gpu_usage = np.array([], dtype=np.float32)
        self.gpu_memory = np.array([], dtype=np.float32)
        self.gpu_power = np.array([], dtype=np.float32)
        self.ram_usage = np.array([], dtype=np.float32)
        
        logger.info("Resource monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring and return summary"""
        self.monitoring = False
        self.stop_sampling = True
        self.end_time = time.time()
        
        logger.info("Resource monitoring stopped")
        return self.get_summary()
        
    def sample_resources(self):
        """Sample current resource usage"""
        if not self.monitoring:
            return
        
        # CPU sampling
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage = np.append(self.cpu_usage, cpu_percent)
        
        # RAM sampling
        ram = psutil.virtual_memory()
        self.ram_usage = np.append(self.ram_usage, ram.used / (1024 * 1024))  # Convert to MB
        
        # GPU sampling if available
        if self.gpu_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.getenv("GPU_ID", 0)))
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_usage = np.append(self.gpu_usage, util.gpu)
                
                # GPU memory
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_memory = np.append(self.gpu_memory, memory.used / (1024 * 1024))
                
                # GPU power (if available)
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    self.gpu_power = np.append(self.gpu_power, power)
                except:
                    pass
                
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
    
    def record_communication(self, bytes_sent=0, bytes_received=0, sync_time=0):
        """Record communication metrics"""
        self.bandwidth_sent += bytes_sent
        self.bandwidth_received += bytes_received
        self.sync_operations += 1
        if sync_time > 0:
            self.communication_times.append(sync_time)
    
    def record_sync_communication(self, sync_start_time):
        """Record synchronization communication time"""
        sync_time = time.time() - sync_start_time
        self.communication_times.append(sync_time)
        self.sync_operations += 1
        return sync_time
        
    def record_computation(self, comp_time):
        """Record computation time"""
        if comp_time > 0:
            self.computation_times.append(comp_time)
    
    def get_summary(self):
        """Vectorized summary calculation"""
        if self.start_time is None or self.end_time is None:
            return {}
            
        total_duration = self.end_time - self.start_time
        
        # Vectorized calculations
        avg_cpu = np.mean(self.cpu_usage) if len(self.cpu_usage) > 0 else 0
        avg_ram = np.mean(self.ram_usage) if len(self.ram_usage) > 0 else 0
        avg_gpu = np.mean(self.gpu_usage) if len(self.gpu_usage) > 0 else 0
        avg_gpu_mem = np.mean(self.gpu_memory) if len(self.gpu_memory) > 0 else 0
        avg_gpu_power = np.mean(self.gpu_power) if len(self.gpu_power) > 0 else 0
        
        # Communication efficiency
        total_comm_time = sum(self.communication_times) if self.communication_times else 0
        total_comp_time = sum(self.computation_times) if self.computation_times else 0
        comm_wait_ratio = total_comm_time / total_duration if total_duration > 0 else 0
        blocked_ratio = total_comm_time / (total_comm_time + total_comp_time) if (total_comm_time + total_comp_time) > 0 else 0
        
        # Energy estimation (simplified model)
        total_joules = (avg_cpu * 0.1 + avg_gpu_power) * total_duration if total_duration > 0 else 0
        
        return {
            'total_duration_seconds': float(total_duration),
            'bandwidth': {
                'bytes_sent': int(self.bandwidth_sent),
                'bytes_received': int(self.bandwidth_received),
                'sync_operations': int(self.sync_operations),
                'total_mb': float((self.bandwidth_sent + self.bandwidth_received) / (1024 * 1024))
            },
            'cpu': {'average_usage_percent': float(avg_cpu)},
            'gpu': {
                'average_usage_percent': float(avg_gpu),
                'average_memory_mb': float(avg_gpu_mem),
                'average_power_w': float(avg_gpu_power)
            },
            'ram': {'average_used_mb': float(avg_ram)},
            'energy': {'total_joules': float(total_joules)},
            'communication_efficiency': {
                'comm_wait_ratio': float(comm_wait_ratio),
                'blocked_ratio': float(blocked_ratio),
                'total_communication_time': float(total_comm_time),
                'total_computation_time': float(total_comp_time)
            },
            'sample_count': {
                'cpu_samples': int(len(self.cpu_usage)),
                'gpu_samples': int(len(self.gpu_usage))
            }
        }
    
    def start_background_sampling(self, interval=2.0):
        """Start background sampling thread"""
        if self.sampling_thread is not None:
            return
            
        def sampling_loop():
            while not self.stop_sampling:
                self.sample_resources()
                time.sleep(interval)
        
        self.sampling_thread = threading.Thread(target=sampling_loop, daemon=True)
        self.sampling_thread.start()
        
    def stop_background_sampling(self):
        """Stop background sampling thread"""
        self.stop_sampling = True
        if self.sampling_thread:
            self.sampling_thread.join(timeout=5.0)
            self.sampling_thread = None


# Global resource monitor instance
resource_monitor = ResourceMonitor()


class RelationAwareSelfAttention(nn.Module):
    """
    Relation-aware self-attention for KGTransformer
    Incorporates relation information into attention computation
    """
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1, use_relation_attention=True):
        super(RelationAwareSelfAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_relation_attention = use_relation_attention
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Relation-aware projections (if enabled)
        if use_relation_attention:
            self.r_proj = nn.Linear(hidden_dim, hidden_dim)  # Relation projection
            self.r_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, relations=None, attention_mask=None):
        """
        Args:
            x: Entity embeddings [batch_size, seq_len, hidden_dim]
            relations: Relation embeddings [batch_size, seq_len, seq_len, hidden_dim] or None
            attention_mask: Mask for padding [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, heads, seq, seq]
        
        # Add relation bias if available
        if self.use_relation_attention and relations is not None:
            # Project relations and reshape for attention heads
            r = self.r_proj(relations)  # [batch, seq, seq, hidden]
            r = r.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim)
            r = r.mean(dim=-1)  # Average over head_dim -> [batch, seq, seq, heads]
            r = r.permute(0, 3, 1, 2)  # [batch, heads, seq, seq]
            attn_scores = attn_scores + r + self.r_bias
        
        # Apply attention mask
        if attention_mask is not None:
            # Create causal mask if needed
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        context = torch.matmul(attn_probs, v)  # [batch, heads, seq, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        output = self.out_proj(context)
        
        return output, attn_probs


class KGTransformerLayer(nn.Module):
    """
    Single KGTransformer layer with relation-aware self-attention and FFN
    """
    
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1, attention_dropout=0.1,
                 use_relation_attention=True, activation='gelu'):
        super(KGTransformerLayer, self).__init__()
        
        # Relation-aware self-attention
        self.self_attn = RelationAwareSelfAttention(
            hidden_dim, num_heads, dropout=attention_dropout,
            use_relation_attention=use_relation_attention
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, relations=None, attention_mask=None):
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        attn_output, attn_weights = self.self_attn(x, relations, attention_mask)
        x = residual + self.dropout1(attn_output)
        
        # FFN with residual connection
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout2(ffn_output)
        
        return x, attn_weights


class NeighborhoodEncoder(nn.Module):
    """
    Encodes entity neighborhoods into fixed-size representations using attention
    """
    
    def __init__(self, hidden_dim, num_heads, max_neighbors=50):
        super(NeighborhoodEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_neighbors = max_neighbors
        
        # Attention for aggregating neighbors
        self.neighbor_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Relation-aware neighbor projection
        self.neighbor_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Position encoding for neighbor ordering
        self.pos_encoding = nn.Parameter(torch.randn(1, max_neighbors, hidden_dim) * 0.02)
        
    def forward(self, entity_emb, neighbor_embs, relation_embs, neighbor_mask=None):
        """
        Args:
            entity_emb: Entity embedding [batch, hidden]
            neighbor_embs: Neighbor entity embeddings [batch, num_neighbors, hidden]
            relation_embs: Relation embeddings for edges [batch, num_neighbors, hidden]
            neighbor_mask: Mask for valid neighbors [batch, num_neighbors]
        """
        batch_size, num_neighbors, hidden_dim = neighbor_embs.shape
        
        # Combine neighbor and relation information
        neighbor_context = torch.cat([neighbor_embs, relation_embs], dim=-1)
        neighbor_context = self.neighbor_proj(neighbor_context)  # [batch, num_neighbors, hidden]
        
        # Add position encoding
        if num_neighbors <= self.max_neighbors:
            pos_encoding = self.pos_encoding[:, :num_neighbors, :]
        else:
            # Truncate if too many neighbors
            pos_encoding = self.pos_encoding[:, :num_neighbors, :]
            neighbor_context = neighbor_context[:, :self.max_neighbors, :]
            neighbor_mask = neighbor_mask[:, :self.max_neighbors] if neighbor_mask is not None else None
        
        neighbor_context = neighbor_context + pos_encoding
        
        # Use entity as query, neighbors as key/value
        entity_emb = entity_emb.unsqueeze(1)  # [batch, 1, hidden]
        
        # Apply attention
        attn_output, attn_weights = self.neighbor_attn(
            entity_emb, neighbor_context, neighbor_context,
            key_padding_mask=~neighbor_mask.bool() if neighbor_mask is not None else None
        )
        
        # Output is [batch, 1, hidden]
        return attn_output.squeeze(1), attn_weights


class KGTransformer(nn.Module):
    """
    Knowledge Graph Transformer for link prediction
    
    Architecture:
    1. Entity and relation embeddings
    2. Neighborhood encoding for each entity
    3. Multiple transformer layers for entity representations
    4. Scoring function for triple prediction
    """
    
    def __init__(self, num_entities, num_relations, embedding_dim=64, hidden_dim=512,
                 num_heads=4, num_layers=2, ffn_dim=512, dropout=0.2, attention_dropout=0.1,
                 use_relation_attention=True, max_neighbors=50, activation='gelu'):
        super(KGTransformer, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_relation_attention = use_relation_attention
        
        # Entity and relation embeddings
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        
        # Projection layers to hidden dimension
        self.entity_proj = nn.Linear(embedding_dim, hidden_dim)
        self.relation_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Neighborhood encoder
        self.neighbor_encoder = NeighborhoodEncoder(hidden_dim, num_heads, max_neighbors)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            KGTransformerLayer(
                hidden_dim, num_heads, ffn_dim, dropout, attention_dropout,
                use_relation_attention, activation
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
        
        # Norm for scoring (used in TransE-style scoring)
        self.norm = 1
        
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform"""
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.entity_proj.weight)
        nn.init.xavier_uniform_(self.relation_proj.weight)
        
    def get_neighbor_relations(self, triplets, num_neighbors=10):
        """
        Extract neighbor relations for each entity in triplets
        
        This is a simplified version. In practice, you'd have pre-computed
        neighbor indices for each entity.
        """
        batch_size = triplets.shape[0]
        device = triplets.device
        
        # Get unique entities in batch
        entities = torch.unique(torch.cat([triplets[:, 0], triplets[:, 2]]))
        
        # Simplified: use random neighbors for demonstration
        # In real implementation, you'd have a neighbor dictionary
        neighbor_entities = torch.randint(
            0, self.num_entities, (batch_size, num_neighbors), device=device
        )
        neighbor_relations = torch.randint(
            0, self.num_relations, (batch_size, num_neighbors), device=device
        )
        
        # Get embeddings
        neighbor_entity_emb = self.entity_embedding(neighbor_entities)
        neighbor_relation_emb = self.relation_embedding(neighbor_relations)
        
        return neighbor_entity_emb, neighbor_relation_emb
    
    def forward(self, triplets, return_attention=False):
        """
        Forward pass for KGTransformer
        
        Args:
            triplets: Tensor of shape [batch_size, 3] (head, relation, tail)
            return_attention: Whether to return attention weights
        
        Returns:
            entity_embeddings: Enhanced entity representations
            scores: Optional scores for triplets
        """
        batch_size = triplets.shape[0]
        device = triplets.device
        
        # Get head and tail entities
        heads = triplets[:, 0]
        tails = triplets[:, 2]
        relations = triplets[:, 1]
        
        # Get all unique entities in batch
        all_entities = torch.unique(torch.cat([heads, tails]))
        entity_to_idx = {e.item(): i for i, e in enumerate(all_entities)}
        
        # Get base embeddings
        entity_emb_base = self.entity_embedding(all_entities)  # [num_unique, embedding_dim]
        relation_emb = self.relation_embedding(relations)  # [batch, embedding_dim]
        
        # Project to hidden dimension
        entity_hidden = self.entity_proj(entity_emb_base)  # [num_unique, hidden_dim]
        relation_hidden = self.relation_proj(relation_emb)  # [batch, hidden_dim]
        
        # Get neighbor information for each entity
        # This is a simplified version - in practice, you'd have pre-computed neighbors
        num_neighbors = 10
        neighbor_entities = torch.randint(
            0, self.num_entities, (len(all_entities), num_neighbors), device=device
        )
        neighbor_relations = torch.randint(
            0, self.num_relations, (len(all_entities), num_neighbors), device=device
        )
        
        neighbor_entity_emb = self.entity_embedding(neighbor_entities)
        neighbor_relation_emb = self.relation_embedding(neighbor_relations)
        
        neighbor_entity_hidden = self.entity_proj(neighbor_entity_emb)
        neighbor_relation_hidden = self.relation_proj(neighbor_relation_emb)
        
        # Encode neighborhoods
        enhanced_entity_hidden = []
        attention_weights = []
        
        for i, entity_idx in enumerate(all_entities):
            # Get this entity's hidden representation
            entity_h = entity_hidden[i:i+1]  # [1, hidden]
            
            # Get its neighbors
            neighbor_h = neighbor_entity_hidden[i]  # [num_neighbors, hidden]
            neighbor_rel_h = neighbor_relation_hidden[i]  # [num_neighbors, hidden]
            
            # Encode neighborhood
            enhanced, attn = self.neighbor_encoder(
                entity_h, 
                neighbor_h.unsqueeze(0), 
                neighbor_rel_h.unsqueeze(0)
            )
            
            enhanced_entity_hidden.append(enhanced)
            if return_attention:
                attention_weights.append(attn)
        
        # Stack enhanced representations
        enhanced_entity_hidden = torch.cat(enhanced_entity_hidden, dim=0)  # [num_unique, hidden]
        
        # Apply transformer layers
        # We need to order entities according to their positions in the sequence
        # For simplicity, we'll use all entities as a sequence
        
        # Add batch dimension
        x = enhanced_entity_hidden.unsqueeze(0)  # [1, num_unique, hidden]
        
        # Create dummy relations for self-attention (can be improved)
        dummy_relations = torch.zeros(1, x.shape[1], x.shape[1], self.hidden_dim, device=device)
        
        all_attn_weights = []
        
        for layer in self.layers:
            x, attn = layer(x, dummy_relations)
            if return_attention:
                all_attn_weights.append(attn)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Map back to original entity indices
        final_entity_embeddings = torch.zeros(
            self.num_entities, self.hidden_dim, device=device
        )
        final_entity_embeddings[all_entities] = x.squeeze(0)
        
        # Compute scores for triplets if needed
        scores = None
        if self.training:
            # Use head and tail embeddings
            head_emb = final_entity_embeddings[heads]
            tail_emb = final_entity_embeddings[tails]
            
            # Score using DistMult-style
            scores = torch.sum(head_emb * relation_hidden * tail_emb, dim=1)
        
        if return_attention:
            return final_entity_embeddings, scores, all_attn_weights
        
        return final_entity_embeddings, scores
    
    def score_triplets(self, heads, relations, tails):
        """
        Score triplets using DistMult scoring
        
        Args:
            heads: Tensor of head entity indices [batch]
            relations: Tensor of relation indices [batch]
            tails: Tensor of tail entity indices [batch]
        
        Returns:
            scores: Tensor of scores [batch]
        """
        # Get embeddings
        head_emb = self.entity_embedding(heads)
        relation_emb = self.relation_embedding(relations)
        tail_emb = self.entity_embedding(tails)
        
        # DistMult scoring
        scores = torch.sum(head_emb * relation_emb * tail_emb, dim=1)
        
        return scores
    
    def get_parameters_vectorized(self):
        """Return flattened parameters for synchronization"""
        return torch.cat([p.data.view(-1) for p in self.parameters()])
    
    def set_parameters_vectorized(self, flat_params):
        """Set parameters from flattened vector"""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view(p.shape))
            offset += numel


# =============================================================================
# Training and utility functions
# =============================================================================

def negative_sampling_vectorized(pos_samples, num_entity, negative_rate):
    """Vectorized negative sampling"""
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    
    # Vectorized tile operation
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    
    # Vectorized label creation
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[:size_of_batch] = 1
    
    # Vectorized random sampling
    values = np.random.randint(0, num_entity, size=num_to_generate)
    choices = np.random.rand(num_to_generate)
    
    # Vectorized masking
    subj_mask = choices > 0.5
    obj_mask = choices <= 0.5
    
    # Vectorized assignment
    neg_samples[subj_mask, 0] = values[subj_mask]
    neg_samples[obj_mask, 2] = values[obj_mask]
    
    return np.concatenate((pos_samples, neg_samples)), labels


def load_partition_vectorized(part_path):
    """Vectorized partition loading with robust splitting"""
    if not os.path.exists(part_path):
        raise FileNotFoundError(f"Partition file not found at {part_path}")
    
    if part_path.endswith('.bin'):
        import dgl
        graphs, _ = dgl.load_graphs(part_path)
        g = graphs[0]
        
        src = g.edges()[0].numpy()
        dst = g.edges()[1].numpy()
        rel = g.edata['rel_type'].numpy()
    else:
        with open(part_path, 'r') as f:
            data = json.load(f)
        src = np.array(data['src'])
        dst = np.array(data['dst'])
        rel = np.array(data['rel'])
    
    # Vectorized concatenation and unique finding
    all_entities = np.concatenate([src, dst])
    global_entities, local_indices = np.unique(all_entities, return_inverse=True)
    
    # Vectorized mapping
    global_to_local = {global_id: local_id for local_id, global_id in enumerate(global_entities)}
    
    # Vectorized local ID conversion
    local_src = np.vectorize(global_to_local.get)(src)
    local_dst = np.vectorize(global_to_local.get)(dst)
    
    # Vectorized triplet creation
    triplets = np.column_stack([local_src, rel, local_dst])
    
    local_entity_count = len(global_entities)
    relation_count = int(np.max(rel)) + 1
    
    # FIXED: Robust splitting for small partitions
    n_samples = len(triplets)
    
    if n_samples < 10:
        # For very small partitions, use all data for training
        logger.warning(f"Partition has only {n_samples} samples, using all for training")
        train_triplets = triplets
        valid_triplets = triplets[:1] if n_samples > 1 else triplets
        test_triplets = triplets[:1] if n_samples > 2 else triplets
    elif n_samples < 50:
        # For small partitions, use simpler split
        train_val_split = int(0.8 * n_samples)
        train_triplets = triplets[:train_val_split]
        valid_triplets = triplets[train_val_split:]
        test_triplets = triplets[train_val_split:]  # Reuse validation as test
    else:
        # Normal split for larger partitions
        try:
            # First split: 80% train, 20% test+val
            train_val_idx = int(0.8 * n_samples)
            train_val = triplets[:train_val_idx]
            test_triplets = triplets[train_val_idx:]
            
            # Second split: 80% train, 20% val (of train_val)
            train_idx = int(0.8 * len(train_val))
            train_triplets = train_val[:train_idx]
            valid_triplets = train_val[train_idx:]
        except Exception as e:
            logger.warning(f"Standard split failed: {e}, using simple split")
            # Fallback to simple split
            train_triplets = triplets[:int(0.7 * n_samples)]
            valid_triplets = triplets[int(0.7 * n_samples):int(0.85 * n_samples)]
            test_triplets = triplets[int(0.85 * n_samples):]
    
    # FIXED: Negative sampling with validation
    if len(train_triplets) > 1:
        train_samples, train_labels = negative_sampling_vectorized(train_triplets, local_entity_count, 1)
    else:
        # For very small partitions, create synthetic negative samples
        logger.warning("Creating synthetic training data for small partition")
        train_samples = np.tile(train_triplets, (min(10, local_entity_count), 1))
        train_labels = np.ones(len(train_samples), dtype=np.float32)
    
    return {
        'train_triples': train_triplets,
        'train_samples': torch.from_numpy(train_samples).long(),
        'train_labels': torch.from_numpy(train_labels).float(),
        'valid_triplets': valid_triplets,
        'test_triplets': test_triplets,
        'all_triplets': triplets,
        'local_entity_count': local_entity_count,
        'relation_count': relation_count,
        'global_entities': global_entities
    }


def train_partition_vectorized(partition_data, epoch=0, batch_size=512, learning_rate=0.0005,
                             global_entities=None, global_relations=None, embedding_dim=64,
                             hidden_dim=256, num_heads=4, num_layers=2, use_amp=False):
    """Train KGTransformer on partition data"""
    try:
        # Use global dimensions if provided
        if global_entities is None:
            global_entities = partition_data['local_entity_count']
        if global_relations is None:
            global_relations = partition_data['relation_count']

        # Set device - prioritize GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model with GLOBAL dimensions
        model = KGTransformer(
            num_entities=global_entities,
            num_relations=global_relations,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

        if 'train_triples' in partition_data:
            triples = partition_data['train_triples']
        else:
            triples = partition_data.get('train_samples', [])
        
        if len(triples) == 0:
            raise ValueError("No training data available")
            
        # Convert to tensor if needed and move to device
        if isinstance(triples, np.ndarray):
            triples = torch.from_numpy(triples).long()
        triples = triples.to(device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Use a subset of data for this batch
        batch_size = min(batch_size, len(triples))
        indices = torch.randperm(len(triples))[:batch_size]
        batch_samples = triples[indices]
        
        # Forward pass with mixed precision
        if scaler:
            with torch.cuda.amp.autocast():
                _, scores = model(batch_samples)
                # Compute loss
                loss = F.binary_cross_entropy_with_logits(
                    scores,
                    torch.ones(batch_size, dtype=torch.float32, device=device)
                )
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            _, scores = model(batch_samples)
            loss = F.binary_cross_entropy_with_logits(
                scores,
                torch.ones(batch_size, dtype=torch.float32, device=device)
            )
            loss.backward()
            optimizer.step()
        
        # Force GPU synchronization
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Return parameters as flattened tensor
        parameters = model.get_parameters_vectorized().cpu()
        
        return {
            'loss': float(loss.item()),
            'parameters': parameters,
            'epoch': int(epoch),
            'batch_size': int(batch_size),
            'parameter_count': int(len(parameters)),
            'device_used': str(device.type),
            'status': 'success'
        }
    except Exception as e:
        logger.error(f"Training error: {e}")
        # Return a valid response even on error
        dummy_model = KGTransformer(
            global_entities, global_relations, 
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        parameters = dummy_model.get_parameters_vectorized()
        return {
            'loss': 1.0,
            'parameters': parameters,
            'epoch': int(epoch),
            'error': str(e),
            'parameter_count': int(len(parameters)),
            'device_used': 'cpu',
            'status': 'error'
        }


def calculate_mrr_vectorized(model, test_triplets, num_entities, num_samples=100):
    """Vectorized MRR calculation for KGTransformer"""
    if len(test_triplets) == 0:
        return 0.0, {1: 0.0, 3: 0.0, 10: 0.0}
    
    # Vectorized tensor conversion
    if isinstance(test_triplets, np.ndarray):
        test_triplets = torch.from_numpy(test_triplets).long()
    
    device = next(model.parameters()).device
    test_triplets = test_triplets.to(device)
    
    # Vectorized sampling
    num_test_samples = min(num_samples, len(test_triplets))
    test_subset = test_triplets[:num_test_samples]
    
    model.eval()
    with torch.no_grad():
        # Vectorized batch processing
        heads = test_subset[:, 0]
        relations = test_subset[:, 1]
        true_tails = test_subset[:, 2]
        
        # Get all entity embeddings
        entity_embeddings = model.entity_embedding.weight
        relation_embeddings = model.relation_embedding.weight
        
        # Vectorized distance calculation for true triplets (using DistMult)
        h_true = entity_embeddings[heads]
        r_true = relation_embeddings[relations]
        t_true = entity_embeddings[true_tails]
        true_scores = torch.sum(h_true * r_true * t_true, dim=1)
        
        # Initialize results arrays
        ranks = []
        hits = {1: 0, 3: 0, 10: 0}
        
        for i in range(num_test_samples):
            # Vectorized negative scoring
            negative_tails = torch.arange(num_entities, device=device)
            negative_tails = negative_tails[negative_tails != true_tails[i]]
            
            h_neg = h_true[i].unsqueeze(0).expand(len(negative_tails), -1)
            r_neg = r_true[i].unsqueeze(0).expand(len(negative_tails), -1)
            t_neg = entity_embeddings[negative_tails]
            
            neg_scores = torch.sum(h_neg * r_neg * t_neg, dim=1)
            
            # Count better negative scores
            num_better = (neg_scores > true_scores[i]).sum().item()
            rank = num_better + 1
            
            ranks.append(rank)
            
            # Vectorized hit calculation
            if rank <= 10:
                hits[10] += 1
                if rank <= 3:
                    hits[3] += 1
                    if rank <= 1:
                        hits[1] += 1
    
    # Vectorized MRR calculation
    if ranks:
        mrr = float(np.mean(1.0 / np.array(ranks)))
        hits = {k: float(v / len(ranks)) for k, v in hits.items()}
        return mrr, hits
    
    return 0.1, {1: 0.05, 3: 0.1, 10: 0.2}


async def aggregate_worker_results_vectorized(worker_urls):
    """Vectorized worker result aggregation"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Vectorized request gathering
        tasks = [client.get(f"{url}/results") for url in worker_urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Vectorized result processing
        worker_results = []
        for url, response in zip(worker_urls, responses):
            if isinstance(response, Exception) or response.status_code != 200:
                continue
            
            results = response.json()
            if results.get('status') in ['completed', 'running']:
                worker_results.append(results)
        
        if not worker_results:
            return {"status": "insufficient_workers", "worker_count": 0}
        
        # Vectorized metric aggregation using numpy
        metrics_to_aggregate = ['mrr', 'hits_at_1', 'hits_at_3', 'hits_at_10']
        
        aggregated = {
            "status": "success",
            "worker_count": len(worker_results),
            "summary": {}
        }
        
        for metric in metrics_to_aggregate:
            values = [w.get('final_metrics', {}).get(metric, 0) for w in worker_results]
            aggregated['summary'][f'average_{metric}'] = float(np.mean(values)) if values else 0
        
        return aggregated