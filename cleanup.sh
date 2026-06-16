#!/bin/bash
# cleanup.sh - Clean up processes and memory

echo "Cleaning up distributed training processes..."

# Kill parameter server
pkill -f "uvicorn parameter_server" || true

# Kill workers
pkill -f "uvicorn worker" || true

# Clear GPU memory
if command -v nvidia-smi &> /dev/null; then
    for i in {0..3}; do
        nvidia-smi -i $i -r -c 0 2>/dev/null || true
    done
fi

# Clear cache
rm -rf logs/*.log 2>/dev/null || true
rm -rf results/*.json 2>/dev/null || true

# Free memory
if [ -w /proc/sys/vm/drop_caches ]; then
    sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
fi

echo "Cleanup complete"
sleep 2