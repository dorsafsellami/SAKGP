# Distributed Knowledge Graph Embedding Framework for WISE conference 2026

This framework enables distributed training of Knowledge Graph Embedding (KGE) models across multiple nodes with comprehensive evaluation of:

- ** Model **: KGTransformer
- **3 Partitioning Methods**: METIS, KaHIP, SAKGP (Semantic-Aware)
- **3 Datasets**: FB15K, FB15K-237, YAGO3-10
- **5 Worker Configurations**: 1, 4, 8, 12, 16 workers (GPUs)
- **Synchronization**: SSP (Stale Synchronous Parallel)



### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPUs (4 per node)
- Grid'5000 access 

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Install KaHIP
git clone https://github.com/KaHIP/KaHIP.git
cd KaHIP
./compile.sh
export PATH=$PATH:$(pwd)/deploy
cd ..



# 2. Run a test experiment
python run_experiment.py \
    --model KGTransformer \
    --dataset FB15K \
    --partition-method metis \
    --num-workers 1 \
    --epochs 10
