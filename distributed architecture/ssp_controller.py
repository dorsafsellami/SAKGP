# ssp_controller.py - Stale Synchronous Parallel Controller
import asyncio
import time
from typing import Dict, Set
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class SSPController:
    """Stale Synchronous Parallel controller for managing worker staleness"""
    
    def __init__(self, max_staleness: int = 2):
        self.max_staleness = max_staleness
        self.worker_versions = defaultdict(int)
        self.worker_clock = defaultdict(float)
        self.barrier_versions = defaultdict(set)
        self.pending_updates = defaultdict(deque)
        self.lock = asyncio.Lock()
        
    async def can_proceed(self, worker_id: str, current_version: int) -> bool:
        """Check if worker can proceed based on SSP rules"""
        async with self.lock:
            if not self.worker_versions:
                return True
                
            min_version = min(self.worker_versions.values())
            staleness = current_version - min_version
            
            if staleness <= self.max_staleness:
                return True
            else:
                logger.debug(f"Worker {worker_id} blocked: staleness {staleness} > max {self.max_staleness}")
                return False
    
    async def update_version(self, worker_id: str, new_version: int):
        """Update worker version"""
        async with self.lock:
            self.worker_versions[worker_id] = new_version
            self.worker_clock[worker_id] = time.time()
            
    async def get_slowest_version(self) -> int:
        """Get the slowest worker's version"""
        async with self.lock:
            return min(self.worker_versions.values()) if self.worker_versions else 0
    
    async def register_barrier(self, version: int, worker_id: str):
        """Register worker at synchronization barrier"""
        async with self.lock:
            self.barrier_versions[version].add(worker_id)
    
    async def is_barrier_ready(self, version: int, total_workers: int) -> bool:
        """Check if all workers have reached the barrier"""
        async with self.lock:
            return len(self.barrier_versions.get(version, set())) >= total_workers
    
    async def cleanup_old_versions(self, current_version: int):
        """Clean up old versions to prevent memory leaks"""
        async with self.lock:
            # Remove versions that are too old
            versions_to_remove = [v for v in self.barrier_versions if v < current_version - 10]
            for v in versions_to_remove:
                del self.barrier_versions[v]
            
            # Remove stale workers (not seen for 5 minutes)
            current_time = time.time()
            workers_to_remove = [wid for wid, last_seen in self.worker_clock.items() 
                               if current_time - last_seen > 300]
            for wid in workers_to_remove:
                del self.worker_versions[wid]
                del self.worker_clock[wid]
    
    async def get_worker_status(self) -> Dict:
        """Get current status of all workers"""
        async with self.lock:
            return {
                "worker_versions": dict(self.worker_versions),
                "slowest_version": min(self.worker_versions.values()) if self.worker_versions else 0,
                "fastest_version": max(self.worker_versions.values()) if self.worker_versions else 0,
                "total_workers": len(self.worker_versions),
                "max_staleness": self.max_staleness
            }
    
    def calculate_communication_efficiency(self):
        """Calculate communication efficiency metrics"""
        if not hasattr(self, 'communication_metrics') or not self.communication_metrics:
            return {}
        
        # Aggregate metrics
        total_bytes_sent = sum(m.get('bytes_sent', 0) for m in self.communication_metrics)
        total_bytes_received = sum(m.get('bytes_received', 0) for m in self.communication_metrics)
        total_sync_ops = sum(m.get('sync_operations', 0) for m in self.communication_metrics)
        
        total_data_mb = (total_bytes_sent + total_bytes_received) / (1024 * 1024)
        
        return {
            'total_data_transferred_mb': float(total_data_mb),
            'total_sync_operations': int(total_sync_ops),
            'avg_transfer_size_mb': float(total_data_mb / total_sync_ops) if total_sync_ops > 0 else 0,
            'bytes_sent_total': int(total_bytes_sent),
            'bytes_received_total': int(total_bytes_received)
        }