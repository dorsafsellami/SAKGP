#!/usr/bin/env python3
# run_experiment.py - Main experiment runner

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from config import (
    get_model_config, get_node_allocation, DATASET_CONFIGS,
    EXPERIMENT_REGISTRY, DEFAULT_SYNC_CONFIG
)

class ExperimentRunner:
    def __init__(self):
        self.results_dir = "results/"
        self.logs_dir = "logs/"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
    def run_partitioning(self, dataset, partition_method, num_workers):
        """Run partitioning step"""
        print(f"\n{'='*60}")
        print(f"PARTITIONING: {dataset} with {partition_method.upper()} into {num_workers} parts")
        print(f"{'='*60}")
        
        script_map = {
            "metis": "metis.py",
            "kahip": "kahip.py",
            "sakgp": "SAKGP.py"
        }
        
        script = script_map[partition_method]
        
        cmd = [
            "python", script,
            "--dataset", dataset,
            "--num-parts", str(num_workers)
        ]
        
        if partition_method == "kahip":
            cmd.extend(["--mode", "strong"])
        
        log_file = f"{self.logs_dir}/partition_{dataset}_{partition_method}_{num_workers}workers.log"
        
        print(f"Running: {' '.join(cmd)}")
        print(f"Log: {log_file}")
        
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        if result.returncode == 0:
            print(f"✓ Partitioning completed successfully")
            return True
        else:
            print(f"✗ Partitioning failed with code {result.returncode}")
            return False
    
    def start_parameter_server(self, port=9000):
        """Start parameter server"""
        print(f"\nStarting Parameter Server on port {port}...")
        
        cmd = [
            "python", "-m", "uvicorn", "parameter_server:app",
            "--host", "0.0.0.0", "--port", str(port)
        ]
        
        log_file = f"{self.logs_dir}/parameter_server.log"
        
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        time.sleep(5)
        print(f"✓ Parameter Server started (PID: {process.pid})")
        return process
    
    def start_worker(self, node_ip, gpu_id, rank, worker_port, ps_url, partition_path):
        """Start a worker process"""
        print(f"  Starting Worker {rank} on {node_ip} (GPU {gpu_id})...")
        
        cmd = [
            "python", "-m", "uvicorn", "worker:app",
            "--host", "0.0.0.0", "--port", str(worker_port)
        ]
        
        env = os.environ.copy()
        env.update({
            "PORT": str(worker_port),
            "RANK": str(rank),
            "WORLD_SIZE": "16",  # Will be updated
            "SERVICE_NAME": "worker",
            "PARTITION_PATH": partition_path,
            "PARAMETER_SERVER_URL": ps_url,
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "GPU_ID": str(gpu_id),
            "COMMUNICATION_STRATEGY": "ssp",
            "SSP_STALENESS": "2",
            "SYNC_INTERVAL": "5",
            "USE_AMP": "True"
        })
        
        log_file = f"{self.logs_dir}/worker_{rank}.log"
        
        with open(log_file, 'w') as f:
            if node_ip.startswith("graffiti") and node_ip != "graffiti-1.nancy.grid5000.fr":
                # Remote node via SSH
                ssh_cmd = ["ssh", node_ip] + cmd
                process = subprocess.Popen(ssh_cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            else:
                # Local node
                process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        
        return process
    
    def initialize_workers(self, worker_ports, node_ip="graffiti-1.nancy.grid5000.fr"):
        """Initialize models on all workers"""
        print("\nInitializing worker models...")
        
        import requests
        
        for rank, port in enumerate(worker_ports):
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"http://{node_ip}:{port}/initialize-model",
                        timeout=30
                    )
                    if response.status_code == 200:
                        print(f"  ✓ Worker {rank} initialized")
                        break
                except Exception as e:
                    print(f"  Worker {rank} init attempt {attempt+1} failed: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(5)
            else:
                print(f"  ✗ Failed to initialize worker {rank}")
    
    def start_training(self, worker_ports, num_epochs, batch_size, node_ip="graffiti-1.nancy.grid5000.fr"):
        """Start training on all workers"""
        print(f"\nStarting training for {num_epochs} epochs...")
        
        import requests
        
        for rank, port in enumerate(worker_ports):
            try:
                response = requests.post(
                    f"http://{node_ip}:{port}/train",
                    params={"num_epochs": num_epochs, "batch_size": batch_size},
                    timeout=10
                )
                if response.status_code == 200:
                    print(f"  ✓ Training started on Worker {rank}")
                else:
                    print(f"  ✗ Failed to start training on Worker {rank}")
            except Exception as e:
                print(f"  ✗ Error starting Worker {rank}: {e}")
    
    def monitor_training(self, worker_ports, node_ip="graffiti-1.nancy.grid5000.fr", timeout=3600):
        """Monitor training progress"""
        print("\nMonitoring training progress...")
        
        import requests
        
        start_time = time.time()
        completed = [False] * len(worker_ports)
        
        while time.time() - start_time < timeout:
            all_completed = True
            
            for rank, port in enumerate(worker_ports):
                if completed[rank]:
                    continue
                    
                try:
                    response = requests.get(f"http://{node_ip}:{port}/status", timeout=5)
                    if response.status_code == 200:
                        status = response.json()
                        if status.get('status') == 'idle' or status.get('status') == 'completed':
                            completed[rank] = True
                            print(f"  ✓ Worker {rank} completed")
                        else:
                            all_completed = False
                            print(f"  Worker {rank}: epoch {status.get('current_epoch', 0)}/{status.get('total_epochs', '?')}")
                except Exception:
                    all_completed = False
            
            if all_completed:
                print("\n✓ All workers completed training")
                return True
            
            time.sleep(30)
        
        print("\n⚠ Training timeout reached")
        return False
    
    def collect_results(self, worker_ports, node_ip="graffiti-1.nancy.grid5000.fr", ps_port=9000):
        """Collect and aggregate results"""
        print("\nCollecting results...")
        
        import requests
        
        # Collect individual worker results
        worker_results = []
        for rank, port in enumerate(worker_ports):
            try:
                response = requests.get(f"http://{node_ip}:{port}/detailed-results", timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    worker_results.append(result)
                    print(f"  ✓ Collected results from Worker {rank}")
            except Exception as e:
                print(f"  ✗ Failed to collect from Worker {rank}: {e}")
        
        # Get aggregated results from parameter server
        try:
            response = requests.get(f"http://{node_ip}:{ps_port}/aggregate-results", timeout=30)
            if response.status_code == 200:
                aggregated = response.json()
                print(f"  ✓ Aggregated results collected")
                return aggregated, worker_results
        except Exception as e:
            print(f"  ✗ Failed to get aggregated results: {e}")
        
        return None, worker_results
    
    def save_experiment_results(self, experiment_config, aggregated_results, worker_results):
        """Save experiment results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result = {
            "experiment_id": timestamp,
            "configuration": experiment_config,
            "timestamp": timestamp,
            "aggregated_results": aggregated_results,
            "worker_results": worker_results
        }
        
        filename = f"{self.results_dir}/exp_{experiment_config['model']}_{experiment_config['dataset']}_{experiment_config['partition_method']}_{experiment_config['num_workers']}workers_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✓ Results saved to {filename}")
        return filename
    
    def run(self, args):
        """Run complete experiment"""
        experiment_id = f"{args.model}_{args.dataset}_{args.partition_method}_{args.num_workers}workers"
        
        print(f"\n{'='*70}")
        print(f" EXPERIMENT: {experiment_id}")
        print(f"{'='*70}")
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Partition Method: {args.partition_method}")
        print(f"Workers: {args.num_workers}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch_size}")
        print(f"{'='*70}\n")
        
        # Get node allocation
        node_allocation = get_node_allocation(args.num_workers)
        worker_ports = [8000 + i for i in range(args.num_workers)]
        
        # Get partition directory
        if args.partition_method == "metis":
            partition_dir = f"data/partitions_metis/{args.dataset}_{args.num_workers}parts"
        elif args.partition_method == "kahip":
            partition_dir = f"data/partitions_kahip/{args.dataset}_{args.num_workers}parts"
        else:  # sakgp
            partition_dir = f"data/partitions_sakgp/{args.dataset}_{args.num_workers}parts"
        
        # Check if partitions exist, run partitioning if not
        if not os.path.exists(partition_dir):
            print(f"\nPartitions not found at {partition_dir}")
            print("Running partitioning step...")
            
            success = self.run_partitioning(args.dataset, args.partition_method, args.num_workers)
            if not success:
                print("✗ Partitioning failed, aborting experiment")
                return
        
        # Start parameter server
        ps_process = self.start_parameter_server(args.ps_port)
        
        # Start workers
        worker_processes = []
        for (node_ip, gpu_id, rank) in node_allocation:
            partition_path = f"{partition_dir}/part-{rank}.bin"
            if not os.path.exists(partition_path):
                print(f"✗ Partition file not found: {partition_path}")
                continue
            
            ps_url = f"http://graffiti-1.nancy.grid5000.fr:{args.ps_port}"
            process = self.start_worker(
                node_ip, gpu_id, rank, worker_ports[rank],
                ps_url, partition_path
            )
            worker_processes.append(process)
            time.sleep(3)
        
        time.sleep(10)
        
        # Initialize workers
        self.initialize_workers(worker_ports)
        
        time.sleep(5)
        
        # Start training
        self.start_training(worker_ports, args.epochs, args.batch_size)
        
        # Monitor training
        self.monitor_training(worker_ports, timeout=args.timeout)
        
        # Collect results
        aggregated, worker_results = self.collect_results(worker_ports, ps_port=args.ps_port)
        
        # Save results
        experiment_config = {
            "model": args.model,
            "dataset": args.dataset,
            "partition_method": args.partition_method,
            "num_workers": args.num_workers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sync_strategy": "ssp",
            "staleness": 2,
            "sync_interval": 5
        }
        
        result_file = self.save_experiment_results(experiment_config, aggregated, worker_results)
        
        # Cleanup
        print("\nCleaning up processes...")
        ps_process.terminate()
        for p in worker_processes:
            p.terminate()
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT COMPLETE: {experiment_id}")
        print(f"Results: {result_file}")
        print(f"{'='*70}\n")
        
        return result_file

def main():
    parser = argparse.ArgumentParser(description="Distributed KGE Experiments")
    
    parser.add_argument("--model", type=str, required=True,
                        choices=["transe", "rgcn", "kgtransformer"],
                        help="Model to train")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["FB15K", "FB15K-237", "YAGO3-10"],
                        help="Dataset to use")
    parser.add_argument("--partition-method", type=str, required=True,
                        choices=["metis", "kahip", "sakgp"],
                        help="Partitioning method")
    parser.add_argument("--num-workers", type=int, required=True,
                        choices=[1, 4, 8, 12, 16],
                        help="Number of workers")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (defaults to model default)")
    parser.add_argument("--ps-port", type=int, default=9000,
                        help="Parameter server port")
    parser.add_argument("--timeout", type=int, default=8000,
                        help="Training timeout in seconds")
    
    args = parser.parse_args()
    
    # Get model config for default batch size
    model_config = get_model_config(args.model, args.dataset)
    if args.batch_size is None:
        args.batch_size = model_config["batch_size"]
    
    # Run experiment
    runner = ExperimentRunner()
    runner.run(args)

if __name__ == "__main__":
    main()