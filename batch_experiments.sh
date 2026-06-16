#!/bin/bash
# batch_experiments.sh - Run all experiments

set -e

RESULTS_DIR="results/"
LOGS_DIR="logs/"
mkdir -p $RESULTS_DIR $LOGS_DIR


echo "=================================================="
echo "Models: TransE, RGCN, KGTransformer"
echo "Datasets: FB15K, FB15K-237, YAGO3-10"
echo "Partition Methods: METIS, KaHIP, SAKGP"
echo "Workers: 1, 4, 8, 12, 16"
echo "=================================================="
echo ""

# Experiment matrix
MODELS=("transe" "rgcn" "kgtransformer")
DATASETS=("FB15K" "FB15K-237" "YAGO3-10")
PARTITION_METHODS=("metis" "kahip" "sakgp")
WORKER_COUNTS=(1 4 8 12 16)

# Optional: limit runs for testing
# WORKER_COUNTS=(1 4)

total=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#PARTITION_METHODS[@]} * ${#WORKER_COUNTS[@]} ))
current=0

echo "Total experiments to run: $total"
echo ""

# Run all experiments
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for partition in "${PARTITION_METHODS[@]}"; do
            for workers in "${WORKER_COUNTS[@]}"; do
                current=$((current + 1))
                
                echo ""
                echo "=================================================="
                echo "[$current/$total] EXPERIMENT: $model - $dataset - $partition - ${workers}workers"
                echo "=================================================="
                
                log_file="${LOGS_DIR}/exp_${model}_${dataset}_${partition}_${workers}workers.log"
                
                # Check if already completed
                if grep -q "EXPERIMENT COMPLETE" "$log_file" 2>/dev/null; then
                    echo "✓ Already completed, skipping..."
                    continue
                fi
                
                # Run experiment
                python run_experiment.py \
                    --model $model \
                    --dataset $dataset \
                    --partition-method $partition \
                    --num-workers $workers \
                    --epochs 200 \
                    --timeout 7200 \
                    | tee "$log_file"
                
                echo "✓ Experiment completed"
                
                # Cool down between experiments
                echo "Cooling down for 30 seconds..."
                sleep 30
            done
        done
    done
done

echo ""
echo "=================================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=================================================="
echo "Results saved in: $RESULTS_DIR"
echo "Logs saved in: $LOGS_DIR"
echo ""

# Generate summary report
python - <<EOF
import json
import os
from pathlib import Path
import pandas as pd

results_dir = "$RESULTS_DIR"
results = []

for f in Path(results_dir).glob("exp_*.json"):
    with open(f, 'r') as fp:
        data = json.load(fp)
    
    config = data['configuration']
    agg = data.get('aggregated_results', {})
    metrics = agg.get('aggregated_metrics', {})
    
    results.append({
        'model': config['model'],
        'dataset': config['dataset'],
        'partition': config['partition_method'],
        'workers': config['num_workers'],
        'mrr': metrics.get('average_mrr', 0),
        'hits@1': metrics.get('average_hits_at_1', 0),
        'hits@3': metrics.get('average_hits_at_3', 0),
        'hits@10': metrics.get('average_hits_at_10', 0),
        'loss': metrics.get('average_loss', 0),
        'speedup': agg.get('scalability', {}).get('speedup', 0),
        'efficiency': agg.get('scalability', {}).get('parallel_efficiency', 0),
    })

if results:
    df = pd.DataFrame(results)
    
    # Pivot tables
    print("\n=== MRR by Model, Dataset, Partition Method (8 workers) ===\n")
    pivot = df[df['workers'] == 8].pivot_table(
        values='mrr', 
        index=['model', 'dataset'], 
        columns='partition',
        aggfunc='mean'
    )
    print(pivot.round(4))
    
    print("\n=== Speedup by Workers ===\n")
    speedup = df.groupby(['model', 'workers'])['speedup'].mean().unstack()
    print(speedup.round(2))
    
    # Save summary
    summary_file = f"{results_dir}/summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
EOF

echo ""
echo "=================================================="