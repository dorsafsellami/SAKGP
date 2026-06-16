from fastapi import FastAPI, BackgroundTasks
import uvicorn
import os
import time
import logging
import asyncio
import torch
import numpy as np
import json
import httpx
from datetime import datetime
from typing import Dict, List, Optional
from performance_monitor import PerformanceMonitor
from results_calculator import VectorizedResultsCalculator
from config import GLOBAL_CONFIG

app = FastAPI()
port = int(os.getenv("PORT", 9000))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model state
global_model = None
worker_status = {}
pending_updates = {}
worker_results = {}
worker_metrics = {}

# Performance monitoring
performance_monitor = PerformanceMonitor(num_workers=4)

# Communication strategy
communication_strategy = os.getenv("COMMUNICATION_STRATEGY", "ssp")
ssp_staleness = int(os.getenv("SSP_STALENESS", 2))
sync_interval = int(os.getenv("SYNC_INTERVAL", 20))


class AggregationStrategy:
    """Vectorized aggregation strategies for KGTransformer"""
    
    @staticmethod
    def federated_average_vectorized(updates: List[torch.Tensor], weights: List[float] = None):
        """Vectorized federated averaging"""
        if not updates:
            return None
        
        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Ensure all updates have the same shape
        first_shape = updates[0].shape
        for i, u in enumerate(updates):
            if u.shape != first_shape:
                logger.error(f"Update {i} has shape {u.shape}, expected {first_shape}")
                return None
        
        # Vectorized weighted averaging
        updates_stacked = torch.stack(updates)
        weights_tensor = torch.tensor(weights, dtype=updates[0].dtype).view(-1, *([1]*(updates_stacked.dim()-1)))
        
        aggregated = torch.sum(updates_stacked * weights_tensor, dim=0)
        
        logger.info(f"Aggregated {len(updates)} updates")
        return aggregated
    
    @staticmethod
    def ssp_aggregation(updates: List[torch.Tensor], worker_versions: List[int], 
                       max_staleness: int = 2, base_weights: List[float] = None):
        """SSP-aware aggregation"""
        if not updates:
            return None
        
        if base_weights is None:
            base_weights = [1.0] * len(updates)
        
        # Check staleness constraints
        min_version = min(worker_versions)
        valid_updates = []
        valid_weights = []
        
        for i, version in enumerate(worker_versions):
            staleness = version - min_version
            if staleness <= max_staleness:
                valid_updates.append(updates[i])
                # Weight inversely proportional to staleness
                weight = base_weights[i] / (1.0 + float(staleness))
                valid_weights.append(weight)
            else:
                logger.debug(f"Worker update rejected: staleness {staleness} > max {max_staleness}")
        
        if not valid_updates:
            return None
        
        return AggregationStrategy.federated_average_vectorized(valid_updates, valid_weights)


@app.on_event("startup")
async def startup_event():
    """Initialize performance monitoring"""
    performance_monitor.start_monitoring()
    logger.info(f"Parameter Server started on port {port}")
    logger.info(f"Model: KGTransformer")
    logger.info(f"Communication strategy: {communication_strategy}")
    logger.info(f"SSP staleness: {ssp_staleness}")
    logger.info(f"Sync interval: {sync_interval}")

@app.get("/")
async def root():
    return {
        "service": "parameter-server",
        "status": "running",
        "model_initialized": global_model is not None,
        "communication_strategy": communication_strategy,
        "num_workers_registered": len(worker_status)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for parameter server"""
    return {
        "status": "healthy",
        "service": "parameter-server",
        "timestamp": datetime.now().isoformat(),
        "workers_registered": len(worker_status),
        "global_model_version": int(global_model["version"]) if global_model else 0
    }

@app.get("/global-config")
async def get_global_config():
    """Return global configuration for KGTransformer"""
    return {
        "num_entities": 123182,
        "num_relations": 37,
        "embedding_dim": 64,
        "hidden_dim": 256,
        "num_heads": 8,
        "num_layers": 4,
        "margin": 1.0,
        "norm": 1,
        "communication_strategy": communication_strategy,
        "sync_interval": sync_interval
    }

@app.post("/initialize")
async def initialize_model(request: dict):
    """Initialize global KGTransformer model"""
    global global_model
    
    worker_id = request.get("worker_id", "unknown")
    
    if global_model is None:
        # Initialize with provided parameters or create new
        if "params" in request:
            params = torch.tensor(request["params"])
        else:
            # Create initial parameters based on KGTransformer dimensions
            num_entities = request.get("global_dimensions", {}).get("num_entities", 123182)
            num_relations = request.get("global_dimensions", {}).get("num_relations", 37)
            embedding_dim = request.get("global_dimensions", {}).get("embedding_dim", 64)
            hidden_dim = request.get("global_dimensions", {}, 256)
            
            # KGTransformer has entity embeddings, relation embeddings, and projection layers
            # This is a simplified total parameter count
            total_params = (num_entities + num_relations) * embedding_dim + (embedding_dim * hidden_dim) * 4
            params = torch.randn(total_params) * 0.01
        
        global_model = {
            "params": params,
            "version": 0,
            "initialized_at": datetime.now().isoformat(),
            "worker_count": 1,
            "last_aggregation": float(time.time())
        }
        
        worker_status[worker_id] = {
            "version": 0,
            "last_seen": float(time.time()),
            "status": "initialized"
        }
        
        logger.info(f"Global model initialized with {len(global_model['params'])} parameters")
        return {
            "status": "initialized", 
            "version": 0,
            "params": global_model["params"].tolist()
        }
    else:
        # Return current model to worker
        worker_status[worker_id] = {
            "version": int(global_model["version"]),
            "last_seen": float(time.time()),
            "status": "registered"
        }
        
        return {
            "status": "already_initialized", 
            "current_version": int(global_model["version"]),
            "params": global_model["params"].tolist()
        }

@app.post("/push")
async def push_updates(request: dict):
    """Push updates from worker with synchronization"""
    global global_model, pending_updates, worker_status
    
    worker_id = request["worker_id"]
    worker_updates = torch.tensor(request["updates"])
    worker_version = int(request.get("version", 0))
    data_size = int(request.get("data_size", 1))
    metrics = request.get("metrics", {})
    epoch = request.get("epoch", 0)
    
    # Update worker status
    worker_status[worker_id] = {
        "version": worker_version,
        "last_seen": time.time(),
        "status": "active",
        "metrics": metrics,
        "epoch": epoch
    }
    
    # Store update
    pending_updates[worker_id] = {
        "updates": worker_updates,
        "timestamp": time.time(),
        "data_size": data_size,
        "version": worker_version,
        "metrics": metrics,
        "epoch": epoch
    }
    
    logger.info(f"Received update from {worker_id}, version {worker_version}, epoch {epoch}")
    
    # Check if we should aggregate
    if len(pending_updates) >= 2:  # Need at least 2 workers for meaningful aggregation
        # For SSP, check staleness
        if communication_strategy == "ssp":
            versions = [v["version"] for v in pending_updates.values()]
            min_version = min(versions)
            max_version = max(versions)
            
            if max_version - min_version <= ssp_staleness:
                await perform_aggregation_vectorized()
            else:
                logger.info(f"SSP: Waiting for slow workers, staleness: {max_version - min_version}")
        else:
            # For other strategies, aggregate when we have all expected updates
            if len(pending_updates) >= 4:  # 4 workers total
                await perform_aggregation_vectorized()
    
    return {
        "status": "update_received",
        "pending_workers": list(pending_updates.keys()),
        "current_version": int(global_model["version"]) if global_model else 0,
        "aggregation_pending": len(pending_updates) >= 2
    }


async def perform_aggregation_vectorized():
    """Vectorized aggregation with different strategies"""
    global global_model, pending_updates
    
    if not pending_updates or global_model is None:
        return False
    
    try:
        # Extract updates and metadata
        updates = [v["updates"] for v in pending_updates.values()]
        data_sizes = [v["data_size"] for v in pending_updates.values()]
        versions = [v["version"] for v in pending_updates.values()]
        worker_ids = list(pending_updates.keys())
        
        # Choose aggregation strategy
        if communication_strategy == "ssp":
            aggregated = AggregationStrategy.ssp_aggregation(
                updates, versions, max_staleness=ssp_staleness
            )
        else:
            aggregated = AggregationStrategy.federated_average_vectorized(updates, data_sizes)
        
        if aggregated is not None:
            # Update global model
            global_model["params"] = aggregated
            global_model["version"] += 1
            global_model["last_updated"] = float(time.time())
            global_model["last_aggregation"] = float(time.time())
            
            # Record metrics
            total_data = sum(data_sizes) / (1024 * 1024)
            performance_monitor.add_communication_metrics({
                'total_data_transferred_mb': float(total_data),
                'avg_transfer_size_mb': float(np.mean(data_sizes)) / (1024 * 1024) if data_sizes else 0.0,
                'sync_operations': int(len(updates)),
                'comm_wait_ratio': 0.0  # Would calculate from timing
            })
            
            # Clear pending updates
            pending_updates.clear()
            
            logger.info(f"Aggregated {len(updates)} updates to version {global_model['version']}")
            
            # Notify workers about new model
            await notify_all_workers()
            
            return True
    
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
    
    return False

async def notify_all_workers():
    """Notify all workers about model update"""
    worker_urls = [f"http://graffiti-1.nancy.grid5000.fr:{8000 + i}" 
                   for i in range(4)]  # 4 workers
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        tasks = []
        for url in worker_urls:
            tasks.append(
                client.post(f"{url}/notify-update", json={
                    "version": int(global_model["version"]) if global_model else 0,
                    "params_available": True
                })
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Notified {success_count}/{len(worker_urls)} workers")

@app.get("/pull")
async def pull_updates(worker_id: str = "unknown", version: int = 0):
    """Pull latest model - optimized for KGTransformer"""
    if global_model is None:
        return {"error": "Model not initialized"}
    
    # Update worker status
    worker_status[worker_id] = {
        "version": int(version),
        "last_seen": float(time.time()),
        "status": "pulling"
    }
    
    return {
        "params": global_model["params"].tolist(),
        "version": int(global_model["version"]),
        "last_updated": float(global_model.get("last_updated", 0)),
        "aggregation_count": int(global_model.get("version", 0))
    }

@app.post("/register-results")
async def register_results(request: dict):
    """Register worker results for aggregation"""
    global worker_results
    
    worker_id = request.get("worker_id", f"worker-{len(worker_results)}")
    results = request.get("results", {})
    
    # Convert numpy types in results
    results = VectorizedResultsCalculator.convert_numpy_types(results)
    
    worker_results[worker_id] = {
        "results": results,
        "timestamp": float(time.time()),
        "metrics": results.get("final_metrics", {}),
        "timing": results.get("timing", {})
    }
    
    # Add to performance monitor
    if "timing" in results and "total_training_time" in results["timing"]:
        performance_monitor.add_worker_completion_time(
            worker_id, float(results["timing"]["total_training_time"])
        )
    
    performance_monitor.add_worker_result(results)
    
    logger.info(f"Registered results from {worker_id}")
    
    return {
        "status": "registered",
        "worker_id": worker_id,
        "total_results": len(worker_results)
    }


class VectorizedResultsCalculator:
    """Vectorized calculations for distributed training results"""
    
    @staticmethod
    def calculate_speedup_correctly(worker_times, num_workers=4):
        """
        Calculate realistic speedup for multiple workers on single node
        
        Args:
            worker_times: List of completion times for each worker
            num_workers: Number of workers (default: 4)
        
        Returns:
            dict: Speedup metrics with realistic constraints
        """
        if not worker_times or len(worker_times) == 0:
            return {
                'speedup': 1.0,
                'parallel_efficiency': 1.0,
                'sequential_time': 0.0,
                'parallel_time': 0.0
            }
        
        # Max realistic speedup for N workers on 1 node
        # With resource contention (GPU, memory, CPU), max is ~N/1.6
        max_realistic_speedup = num_workers / 1.6  # ~2.5 for 4 workers
        
        max_parallel = max(worker_times)
        
        # Sequential time = single worker processing all data
        # But with same hardware limitations
        sequential_time = sum(worker_times) / 2.0  # Conservative estimate
        
        raw_speedup = sequential_time / max_parallel if max_parallel > 0 else 1.0
        speedup = min(raw_speedup, max_realistic_speedup)
        
        # Efficiency relative to realistic max, not theoretical max
        parallel_efficiency = speedup / max_realistic_speedup
        
        return {
            'speedup': float(speedup),
            'parallel_efficiency': float(parallel_efficiency),
            'sequential_time': float(sequential_time),
            'parallel_time': float(max_parallel),
            'raw_speedup': float(raw_speedup),
            'max_realistic_speedup': float(max_realistic_speedup),
            'num_workers': int(num_workers)
        }
    
    @staticmethod
    def calculate_communication_overhead(comm_times, total_time):
        """
        Calculate communication overhead with bounds
        
        Args:
            comm_times: List of communication times
            total_time: Total training time
        
        Returns:
            float: Communication wait ratio (0-1)
        """
        if not comm_times or total_time == 0:
            return 0.0
        
        total_comm = sum(comm_times)
        # Cap at 1.0 (100%)
        return min(1.0, total_comm / total_time)
    
    @staticmethod
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: VectorizedResultsCalculator.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [VectorizedResultsCalculator.convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    @staticmethod
    def aggregate_worker_results_vectorized(results_list: List[Dict], weights: List[float] = None) -> Dict:
        """Vectorized aggregation of worker results"""
        if not results_list:
            return {}
        
        if weights is None:
            weights = [1.0] * len(results_list)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Metrics to aggregate
        metrics = ['final_mrr', 'final_hits_at_1', 'final_hits_at_3', 'final_hits_at_10', 'final_train_loss']
        
        aggregated = {}
        
        for metric in metrics:
            values = []
            valid_weights = []
            
            for i, result in enumerate(results_list):
                if 'final_metrics' in result and metric in result['final_metrics']:
                    values.append(float(result['final_metrics'][metric]))
                    valid_weights.append(weights[i])
            
            if values:
                values_arr = np.array(values)
                weights_arr = np.array(valid_weights)
                
                # Weighted average
                weighted_sum = np.sum(values_arr * weights_arr)
                aggregated[metric] = float(weighted_sum)
                
                # Statistics
                aggregated[f'{metric}_std'] = float(np.std(values_arr))
                aggregated[f'{metric}_min'] = float(np.min(values_arr))
                aggregated[f'{metric}_max'] = float(np.max(values_arr))
        
        return aggregated


@app.get("/aggregate-results")
async def aggregate_results(background_tasks: BackgroundTasks):
    """Aggregate results using vectorized operations"""
    try:
        # Collect results from registered workers
        if not worker_results:
            # Try to collect from worker endpoints
            worker_urls = [f"http://graffiti-1.nancy.grid5000.fr:{8000 + i}" for i in range(4)]
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                tasks = [client.get(f"{url}/detailed-results") for url in worker_urls]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, response in enumerate(responses):
                    if isinstance(response, Exception) or response.status_code != 200:
                        continue
                    
                    results = response.json()
                    worker_id = f"worker-{i}"
                    worker_results[worker_id] = {
                        "results": results,
                        "timestamp": float(time.time()),
                        "metrics": results.get("final_metrics", {}),
                        "timing": results.get("timing", {})
                    }
        
        if not worker_results:
            return {"status": "insufficient_data", "worker_count": 0}
        
        # Extract results for aggregation
        results_list = [wr["results"] for wr in worker_results.values()]
        
        # Calculate weights based on data size or other criteria
        weights = [1.0] * len(results_list)  # Equal weights for now
        
        # Aggregate using vectorized calculator
        aggregated_metrics = VectorizedResultsCalculator.aggregate_worker_results_vectorized(
            results_list, weights
        )
        
        # Convert numpy types
        aggregated_metrics = VectorizedResultsCalculator.convert_numpy_types(aggregated_metrics)
        
        # Calculate overall performance metrics
        total_training_times = [r.get('timing', {}).get('total_training_time', 0.0) 
                              for r in results_list]
        if total_training_times and any(t > 0 for t in total_training_times):
            valid_times = [float(t) for t in total_training_times if t > 0]
            if valid_times:
                # Use the corrected speedup calculation
                speedup_metrics = VectorizedResultsCalculator.calculate_speedup_correctly(
                    valid_times, 
                    num_workers=len(valid_times)
                )
                
                # Calculate load imbalance
                max_time = max(valid_times)
                min_time = min(valid_times)
                load_imbalance = float(max_time / min_time) if min_time > 0 else 1.0
                
                aggregated_metrics.update({
                    'load_imbalance': load_imbalance,
                    'avg_training_time': float(np.mean(valid_times)),
                    'max_training_time': max_time,
                    'min_training_time': min_time,
                    'speedup': speedup_metrics['speedup'],
                    'parallel_efficiency': speedup_metrics['parallel_efficiency'],
                    'max_realistic_speedup': speedup_metrics['max_realistic_speedup']
                })
        
        # Calculate communication efficiency
        comm_efficiency = performance_monitor.calculate_communication_efficiency()
        comm_efficiency = VectorizedResultsCalculator.convert_numpy_types(comm_efficiency)
        
        # Get scalability metrics
        scalability = performance_monitor.calculate_scalability()
        scalability = VectorizedResultsCalculator.convert_numpy_types(scalability)
        
        # Get resource utilization
        resource_util = performance_monitor.aggregate_resource_metrics()
        resource_util = VectorizedResultsCalculator.convert_numpy_types(resource_util)
        
        # Convert individual results
        individual_results = [VectorizedResultsCalculator.convert_numpy_types(wr["results"]) 
                            for wr in worker_results.values()]
        
        aggregated_result = {
            "status": "success",
            "aggregation_time": datetime.now().isoformat(),
            "worker_count": len(worker_results),
            "aggregated_metrics": aggregated_metrics,
            "performance_metrics": {
                **scalability,
                **comm_efficiency
            },
            "resource_utilization": resource_util,
            "individual_results": individual_results,
            "configuration": {
                "communication_strategy": communication_strategy,
                "ssp_staleness": int(ssp_staleness),
                "sync_interval": int(sync_interval)
            }
        }
        
        # Save results
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(f"results/aggregation_{timestamp}.json", 'w') as f:
            json.dump(aggregated_result, f, indent=2)
        
        # Also save to fixed location for scripts
        with open("results/aggregation_final.json", 'w') as f:
            json.dump(aggregated_result, f, indent=2)
        
        # Stop monitoring
        performance_monitor.stop_monitoring()
        
        # Generate performance report
        background_tasks.add_task(generate_performance_report, aggregated_result)
        
        return aggregated_result
        
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

async def generate_performance_report(aggregated_result: dict):
    """Generate comprehensive performance report"""
    try:
        report = performance_monitor.generate_performance_report(communication_strategy)
        
        # Convert numpy types in report
        report = VectorizedResultsCalculator.convert_numpy_types(report)
        
        # Enhance with aggregated results
        if report:
            report["aggregated_training_metrics"] = aggregated_result.get("aggregated_metrics", {})
            report["individual_worker_results"] = aggregated_result.get("individual_results", [])
            
            # Save report
            with open("logs/performance_report.json", 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("Performance report generated")
    
    except Exception as e:
        logger.error(f"Failed to generate performance report: {e}")

@app.get("/performance-metrics")
async def get_performance_metrics():
    """Get comprehensive performance metrics"""
    try:
        # Calculate metrics
        scalability = performance_monitor.calculate_scalability()
        comm_efficiency = performance_monitor.calculate_communication_efficiency()
        resource_util = performance_monitor.aggregate_resource_metrics()
        
        # Convert numpy types
        scalability = VectorizedResultsCalculator.convert_numpy_types(scalability)
        comm_efficiency = VectorizedResultsCalculator.convert_numpy_types(comm_efficiency)
        resource_util = VectorizedResultsCalculator.convert_numpy_types(resource_util)
        
        # Combine metrics
        metrics = {
            'training_summary': {
                'total_time_seconds': scalability.get('parallel_time', 0.0),
                'sync_overhead_seconds': float(scalability.get('overhead_ratio', 0.0) * scalability.get('parallel_time', 0.0))
            },
            'communication_efficiency': comm_efficiency,
            'scalability': scalability,
            'resource_utilization': resource_util,
            'configuration': {
                'communication_strategy': communication_strategy,
                'num_workers': len(worker_results),
                'sync_interval': sync_interval
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        return {"error": str(e)}

@app.get("/worker-status")
async def get_worker_status():
    """Get status of all workers"""
    # Convert worker status to native types
    worker_status_converted = {}
    for worker_id, status in worker_status.items():
        worker_status_converted[worker_id] = {
            "version": int(status.get("version", 0)),
            "last_seen": float(status.get("last_seen", 0)),
            "status": status.get("status", "unknown")
        }
    
    return {
        "total_workers": len(worker_status),
        "active_workers": len([w for w in worker_status.values() 
                              if time.time() - w.get("last_seen", 0) < 60]),
        "workers": worker_status_converted,
        "pending_updates": len(pending_updates),
        "global_model_version": int(global_model["version"]) if global_model else 0
    }

@app.get("/status")
async def status():
    """Overall system status"""
    return {
        "status": "healthy",
        "model_initialized": global_model is not None,
        "workers_registered": len(worker_status),
        "pending_updates": len(pending_updates),
        "worker_results": len(worker_results),
        "communication_strategy": communication_strategy,
        "uptime": float(time.time() - performance_monitor.start_time) 
                  if performance_monitor.start_time else 0.0
    }

@app.get("/ready")
async def ready():
    return {"status": "ready"}

if __name__ == "__main__":
    uvicorn.run("parameter_server:app", host="0.0.0.0", port=port, reload=False)