# performance_monitor.py - Enhanced with KGTransformer specific metrics
import numpy as np
import time
import json
from typing import Dict, List
from datetime import datetime

class PerformanceMonitor:
    """Comprehensive performance monitoring for distributed KGTransformer training"""
    
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.worker_completion_times = []
        self.worker_results = []
        self.communication_metrics = []
        self.transformer_metrics = {
            'mrr_progression': [],
            'hits_progression': {1: [], 3: [], 10: []},
            'loss_progression': []
        }
        self.start_time = None
        self.end_time = None
        
    def start_monitoring(self):
        self.start_time = float(time.time())
        
    def stop_monitoring(self):
        self.end_time = float(time.time())
        
    def add_worker_completion_time(self, worker_id: str, completion_time: float):
        self.worker_completion_times.append((worker_id, float(completion_time)))
        
    def add_worker_result(self, worker_result: Dict):
        self.worker_results.append(worker_result)
        
        # Extract KGTransformer specific metrics
        if 'final_metrics' in worker_result:
            metrics = worker_result['final_metrics']
            if 'final_mrr' in metrics:
                self.transformer_metrics['mrr_progression'].append(float(metrics['final_mrr']))
            if 'final_train_loss' in metrics:
                self.transformer_metrics['loss_progression'].append(float(metrics['final_train_loss']))
            for k in [1, 3, 10]:
                key = f'final_hits_at_{k}'
                if key in metrics:
                    self.transformer_metrics['hits_progression'][k].append(float(metrics[key]))
        
    def add_communication_metrics(self, metrics: Dict):
        # Convert numpy types in metrics
        converted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.int32, np.int64)):
                converted_metrics[key] = int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                converted_metrics[key] = float(value)
            elif isinstance(value, np.bool_):
                converted_metrics[key] = bool(value)
            elif isinstance(value, np.ndarray):
                converted_metrics[key] = value.tolist()
            else:
                converted_metrics[key] = value
        
        self.communication_metrics.append(converted_metrics)
        
    def calculate_communication_efficiency(self):
        """Calculate communication efficiency metrics from recorded data"""
        if not self.communication_metrics:
            return {}
        
        # Aggregate metrics
        total_bytes_sent = 0
        total_bytes_received = 0
        total_sync_ops = 0
        total_comm_time = 0
        
        for metrics in self.communication_metrics:
            total_bytes_sent += metrics.get('bytes_sent', 0)
            total_bytes_received += metrics.get('bytes_received', 0)
            total_sync_ops += metrics.get('sync_operations', 0)
            total_comm_time += metrics.get('comm_time', 0)
        
        total_data_mb = (total_bytes_sent + total_bytes_received) / (1024 * 1024)
        total_time = (self.end_time - self.start_time) if self.end_time and self.start_time else 1.0
        
        comm_wait_ratio = min(1.0, total_comm_time / total_time) if total_time > 0 else 0.0
        
        return {
            'total_data_transferred_mb': float(total_data_mb),
            'total_sync_operations': int(total_sync_ops),
            'avg_transfer_size_mb': float(total_data_mb / total_sync_ops) if total_sync_ops > 0 else 0.0,
            'bytes_sent_total': int(total_bytes_sent),
            'bytes_received_total': int(total_bytes_received),
            'comm_wait_ratio': float(comm_wait_ratio),
            'communication_efficiency': float(1.0 - comm_wait_ratio)
        }

    def calculate_scalability(self, sequential_time: float = None):
        """Calculate scalability metrics with realistic constraints"""
        if not self.worker_completion_times:
            return {}
            
        times = [float(t) for _, t in self.worker_completion_times]
        
        # Basic statistics
        max_time = max(times)
        min_time = min(times)
        avg_time = float(np.mean(times))
        std_time = float(np.std(times))
        
        # Load imbalance
        load_imbalance = float(max_time / min_time) if min_time > 0 else 1.0
        
        # Use corrected speedup calculation
        from parameter_server import VectorizedResultsCalculator
        
        speedup_metrics = VectorizedResultsCalculator.calculate_speedup_correctly(
            times, 
            num_workers=self.num_workers
        )
        
        # Workload imbalance
        workload_imbalance = float(std_time / avg_time) if avg_time > 0 else 0.0
        
        # Overhead ratio
        total_time_workers = sum(times)
        overhead_ratio = float((total_time_workers - speedup_metrics['sequential_time']) / 
                            speedup_metrics['sequential_time']) if speedup_metrics['sequential_time'] > 0 else 0.0
        overhead_ratio = max(0.0, min(overhead_ratio, 2.0))
        
        # Scalability score (0-1)
        scalability_score = float(speedup_metrics['parallel_efficiency'] * (1 - workload_imbalance))
        
        return {
            'load_imbalance_ratio': float(load_imbalance),
            'workload_imbalance': float(workload_imbalance),
            'max_completion_time': float(max_time),
            'min_completion_time': float(min_time),
            'avg_completion_time': float(avg_time),
            'std_dev_completion': float(std_time),
            'num_workers': int(self.num_workers),
            'parallel_efficiency': float(speedup_metrics['parallel_efficiency']),
            'speedup': float(speedup_metrics['speedup']),
            'scalability_score': float(scalability_score),
            'overhead_ratio': float(overhead_ratio),
            'sequential_time': float(speedup_metrics['sequential_time']),
            'parallel_time': float(max_time),
            'max_realistic_speedup': float(speedup_metrics['max_realistic_speedup'])
        }
    
    def aggregate_resource_metrics(self):
        """Aggregate resource utilization metrics from all workers"""
        if not self.worker_results:
            return {}
            
        # Extract resource metrics from worker results
        cpu_usage = []
        gpu_usage = []
        gpu_memory = []
        gpu_power = []
        ram_usage = []
        energies = []
        bandwidth_sent = 0
        bandwidth_received = 0
        sync_ops = 0
        
        for result in self.worker_results:
            if 'resource_monitoring' in result:
                rm = result['resource_monitoring']
                
                if 'cpu' in rm:
                    cpu_usage.append(float(rm['cpu'].get('average_usage_percent', 0.0)))
                if 'gpu' in rm:
                    gpu_usage.append(float(rm['gpu'].get('average_usage_percent', 0.0)))
                    gpu_memory.append(float(rm['gpu'].get('average_memory_mb', 0.0)))
                    gpu_power.append(float(rm['gpu'].get('average_power_w', 0.0)))
                if 'ram' in rm:
                    ram_usage.append(float(rm['ram'].get('average_used_mb', 0.0)))
                if 'energy' in rm:
                    energies.append(float(rm['energy'].get('total_joules', 0.0)))
                if 'bandwidth' in rm:
                    bandwidth_sent += int(rm['bandwidth'].get('bytes_sent', 0))
                    bandwidth_received += int(rm['bandwidth'].get('bytes_received', 0))
                    sync_ops += int(rm['bandwidth'].get('sync_operations', 0))
        
        # Calculate averages
        avg_cpu = float(np.mean(cpu_usage)) if cpu_usage else 0.0
        avg_gpu = float(np.mean(gpu_usage)) if gpu_usage else 0.0
        avg_gpu_memory = float(np.mean(gpu_memory)) if gpu_memory else 0.0
        avg_gpu_power = float(np.mean(gpu_power)) if gpu_power else 0.0
        avg_ram = float(np.mean(ram_usage)) if ram_usage else 0.0
        total_energy = float(sum(energies)) if energies else 0.0
        total_data_mb = float(bandwidth_sent + bandwidth_received) / (1024 * 1024)
        
        # Resource utilization efficiency
        cpu_efficiency = avg_cpu / 100.0
        gpu_efficiency = avg_gpu / 100.0
        overall_resource_efficiency = float((cpu_efficiency + gpu_efficiency) / 2.0)
        
        return {
            'bandwidth': {
                'bytes_sent': int(bandwidth_sent),
                'bytes_received': int(bandwidth_received),
                'sync_operations': int(sync_ops),
                'total_mb': float(total_data_mb),
                'avg_sync_data_mb': float(total_data_mb / sync_ops) if sync_ops > 0 else 0.0
            },
            'cpu': {
                'average_usage_percent': float(avg_cpu),
                'efficiency': float(cpu_efficiency)
            },
            'gpu': {
                'average_usage_percent': float(avg_gpu),
                'average_memory_mb': float(avg_gpu_memory),
                'average_power_w': float(avg_gpu_power),
                'efficiency': float(gpu_efficiency)
            },
            'ram': {'average_used_mb': float(avg_ram)},
            'energy': {'total_joules': float(total_energy)},
            'resource_efficiency': float(overall_resource_efficiency)
        }
    
    def calculate_transformer_performance(self):
        """Calculate KGTransformer specific performance metrics"""
        if not self.transformer_metrics['mrr_progression']:
            return {}
        
        mrr_values = self.transformer_metrics['mrr_progression']
        loss_values = self.transformer_metrics['loss_progression']
        
        # Convergence metrics
        if len(loss_values) > 1:
            loss_improvement = loss_values[0] - loss_values[-1]
            convergence_rate = float(loss_improvement / len(loss_values)) if len(loss_values) > 0 else 0.0
        else:
            loss_improvement = 0.0
            convergence_rate = 0.0
        
        # MRR improvement
        if len(mrr_values) > 1:
            mrr_improvement = mrr_values[-1] - mrr_values[0]
            mrr_learning_rate = float(mrr_improvement / len(mrr_values)) if len(mrr_values) > 0 else 0.0
        else:
            mrr_improvement = 0.0
            mrr_learning_rate = 0.0
        
        # Hits metrics
        hits_metrics = {}
        for k in [1, 3, 10]:
            hits = self.transformer_metrics['hits_progression'].get(k, [])
            if hits:
                hits_metrics[f'hits_at_{k}'] = {
                    'final': float(hits[-1]),
                    'average': float(np.mean(hits)),
                    'improvement': float(hits[-1] - hits[0]) if len(hits) > 1 else 0.0
                }
        
        return {
            'mrr': {
                'final': float(mrr_values[-1]) if mrr_values else 0.0,
                'average': float(np.mean(mrr_values)) if mrr_values else 0.0,
                'improvement': float(mrr_improvement),
                'learning_rate': float(mrr_learning_rate),
                'max': float(max(mrr_values)) if mrr_values else 0.0,
                'min': float(min(mrr_values)) if mrr_values else 0.0
            },
            'loss': {
                'final': float(loss_values[-1]) if loss_values else 0.0,
                'average': float(np.mean(loss_values)) if loss_values else 0.0,
                'improvement': float(loss_improvement),
                'convergence_rate': float(convergence_rate)
            },
            'hits': hits_metrics,
            'quality_score': float((mrr_values[-1] if mrr_values else 0.0) * 0.4 + 
                            sum([h['final'] for h in hits_metrics.values()]) / len(hits_metrics) * 0.6 
                            if hits_metrics else 0.0)
        }
    
    def generate_performance_report(self, communication_strategy: str):
        """Generate comprehensive performance report for KGTransformer"""
        if not self.start_time or not self.end_time:
            return {}
            
        total_time = self.end_time - self.start_time
        
        # Calculate all metrics
        scalability = self.calculate_scalability()
        comm_efficiency = self.calculate_communication_efficiency()
        resource_util = self.aggregate_resource_metrics()
        transformer_performance = self.calculate_transformer_performance()
        
        # Aggregate training metrics
        aggregated_metrics = {}
        if self.worker_results:
            # Extract MRR and hits
            mrr_values = []
            hits1_values = []
            hits3_values = []
            hits10_values = []
            
            for result in self.worker_results:
                if 'final_metrics' in result:
                    metrics = result['final_metrics']
                    mrr_values.append(float(metrics.get('final_mrr', 0.0)))
                    hits1_values.append(float(metrics.get('final_hits_at_1', 0.0)))
                    hits3_values.append(float(metrics.get('final_hits_at_3', 0.0)))
                    hits10_values.append(float(metrics.get('final_hits_at_10', 0.0)))
            
            if mrr_values:
                aggregated_metrics = {
                    'average_mrr': float(np.mean(mrr_values)),
                    'average_hits_at_1': float(np.mean(hits1_values)) if hits1_values else 0.0,
                    'average_hits_at_3': float(np.mean(hits3_values)) if hits3_values else 0.0,
                    'average_hits_at_10': float(np.mean(hits10_values)) if hits10_values else 0.0
                }
        
        # Overall performance score
        overall_score = float(
            scalability.get('scalability_score', 0.0) * 0.3 +
            comm_efficiency.get('communication_efficiency', 0.0) * 0.2 +
            resource_util.get('resource_efficiency', 0.0) * 0.2 +
            transformer_performance.get('quality_score', 0.0) * 0.3
        )
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'communication_strategy': communication_strategy,
            'training_summary': {
                'total_time_seconds': float(total_time),
                'num_workers': int(self.num_workers),
                'successful_workers': int(len(self.worker_results)),
                'overall_performance_score': float(overall_score)
            },
            'scalability': scalability,
            'communication_efficiency': comm_efficiency,
            'resource_utilization': resource_util,
            'transformer_performance': transformer_performance,
            'aggregated_metrics': aggregated_metrics,
            'recommendations': self.generate_recommendations(
                scalability, comm_efficiency, resource_util, transformer_performance
            )
        }
        
        return report
    
    def generate_recommendations(self, scalability, comm_efficiency, resource_util, transformer_performance):
        """Generate recommendations based on performance metrics"""
        recommendations = []
        
        # Scalability recommendations
        if scalability.get('parallel_efficiency', 0.0) < 0.7:
            recommendations.append({
                'area': 'scalability',
                'issue': 'Low parallel efficiency',
                'suggestion': 'Consider increasing batch size or adjusting synchronization frequency'
            })
        
        if scalability.get('load_imbalance_ratio', 1.0) > 1.2:
            recommendations.append({
                'area': 'scalability',
                'issue': 'High load imbalance',
                'suggestion': 'Re-partition data more evenly across workers'
            })
        
        # Communication recommendations
        if comm_efficiency.get('comm_wait_ratio', 0.0) > 0.3:
            recommendations.append({
                'area': 'communication',
                'issue': 'High communication wait time',
                'suggestion': 'Reduce synchronization frequency or increase staleness tolerance'
            })
        
        # Resource utilization recommendations
        if resource_util.get('gpu', {}).get('average_usage_percent', 0.0) < 50.0:
            recommendations.append({
                'area': 'resource',
                'issue': 'Low GPU utilization',
                'suggestion': 'Increase batch size or enable mixed precision training'
            })
        
        # Transformer performance recommendations
        if transformer_performance.get('mrr', {}).get('final', 0.0) < 0.2:
            recommendations.append({
                'area': 'model',
                'issue': 'Low MRR score',
                'suggestion': 'Consider increasing embedding dimension or adjusting learning rate'
            })
        
        if transformer_performance.get('loss', {}).get('convergence_rate', 0.0) < 0.01:
            recommendations.append({
                'area': 'model',
                'issue': 'Slow convergence',
                'suggestion': 'Adjust learning rate or try different optimization algorithm'
            })
        
        return recommendations