# results_calculator.py - FIXED for JSON serialization
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import time

class VectorizedResultsCalculator:
    """Vectorized calculation of training results and metrics"""
    
    @staticmethod
    def calculate_metrics_vectorized(
        train_losses: List[float],
        val_losses: List[float] = None,
        mrrs: List[float] = None,
        hits_dict: Dict[int, List[float]] = None
    ) -> Dict:
        """Calculate all metrics using vectorized operations"""
        
        # Convert to numpy arrays for vectorized operations
        train_loss_arr = np.array(train_losses) if train_losses else np.array([])
        val_loss_arr = np.array(val_losses) if val_losses else np.array([])
        mrr_arr = np.array(mrrs) if mrrs else np.array([])
        
        # Calculate averages using vectorized operations
        metrics = {}
        
        if len(train_loss_arr) > 0:
            metrics.update({
                'final_train_loss': float(train_loss_arr[-1]),
                'avg_train_loss': float(np.mean(train_loss_arr)),
                'min_train_loss': float(np.min(train_loss_arr)),
                'max_train_loss': float(np.max(train_loss_arr)),
                'std_train_loss': float(np.std(train_loss_arr)),
                'train_loss_trend': float(train_loss_arr[-1] - train_loss_arr[0]) if len(train_loss_arr) > 1 else 0.0
            })
        
        if len(val_loss_arr) > 0:
            metrics.update({
                'final_val_loss': float(val_loss_arr[-1]),
                'avg_val_loss': float(np.mean(val_loss_arr)),
                'min_val_loss': float(np.min(val_loss_arr))
            })
        
        if len(mrr_arr) > 0:
            metrics.update({
                'final_mrr': float(mrr_arr[-1]),
                'avg_mrr': float(np.mean(mrr_arr)),
                'max_mrr': float(np.max(mrr_arr)),
                'mrr_improvement': float(mrr_arr[-1] - mrr_arr[0]) if len(mrr_arr) > 1 else 0.0
            })
        
        # Calculate hits metrics
        if hits_dict:
            for k in [1, 3, 10]:
                if k in hits_dict and hits_dict[k]:
                    hits_arr = np.array(hits_dict[k])
                    metrics.update({
                        f'final_hits_at_{k}': float(hits_arr[-1]),
                        f'avg_hits_at_{k}': float(np.mean(hits_arr)),
                        f'max_hits_at_{k}': float(np.max(hits_arr))
                    })
        
        return metrics
    
    @staticmethod
    def calculate_timing_metrics_vectorized(
        epoch_times: List[float],
        total_training_time: float = None
    ) -> Dict:
        """Calculate timing metrics using vectorized operations"""
        
        epoch_times_arr = np.array(epoch_times) if epoch_times else np.array([])
        
        timing_metrics = {}
        
        if len(epoch_times_arr) > 0:
            timing_metrics.update({
                'total_training_time': float(np.sum(epoch_times_arr)) if total_training_time is None else float(total_training_time),
                'avg_epoch_time': float(np.mean(epoch_times_arr)),
                'min_epoch_time': float(np.min(epoch_times_arr)),
                'max_epoch_time': float(np.max(epoch_times_arr)),
                'std_epoch_time': float(np.std(epoch_times_arr)),
                'total_epochs': int(len(epoch_times_arr)),
                'throughput_epochs_per_second': float(len(epoch_times_arr) / np.sum(epoch_times_arr)) if np.sum(epoch_times_arr) > 0 else 0.0
            })
        
        return timing_metrics
    
    @staticmethod
    def calculate_convergence_metrics_vectorized(
        losses: List[float],
        mrrs: List[float] = None,
        threshold: float = 1e-4
    ) -> Dict:
        """Calculate convergence metrics using vectorized operations"""
        
        losses_arr = np.array(losses) if losses else np.array([])
        
        if len(losses_arr) < 2:
            return {}
        
        # Calculate gradients using vectorized difference
        gradients = np.diff(losses_arr)
        
        # Find convergence point (where gradient magnitude < threshold)
        convergence_idx = np.where(np.abs(gradients) < threshold)[0]
        
        convergence_epoch = int(convergence_idx[0] + 1) if len(convergence_idx) > 0 else len(losses_arr)
        
        metrics = {
            'convergence_epoch': convergence_epoch,
            'final_gradient': float(gradients[-1]) if len(gradients) > 0 else 0.0,
            'avg_gradient': float(np.mean(np.abs(gradients))) if len(gradients) > 0 else 0.0,
            'has_converged': bool(convergence_epoch < len(losses_arr))
        }
        
        # Calculate MRR at convergence if available
        if mrrs and len(mrrs) > convergence_epoch:
            mrr_arr = np.array(mrrs)
            metrics['mrr_at_convergence'] = float(mrr_arr[convergence_epoch - 1])
        
        return metrics
    
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
    
    @staticmethod
    def convert_numpy_types(obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization"""
        if obj is None:
            return None
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {str(k): VectorizedResultsCalculator.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [VectorizedResultsCalculator.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(VectorizedResultsCalculator.convert_numpy_types(item) for item in obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj