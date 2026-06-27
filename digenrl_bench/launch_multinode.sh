#!/bin/bash
# Launch a torchrun job across the 4 zw1 nodes via clusterbridge.
# Usage: bash launch_multinode.sh <script.py> <logtag> [extra args...]
set -u
CB=/root/shared/.clusters/.tools/clusterbridge.sh
SCRIPT="$1"; TAG="$2"; shift 2; EXTRA="$*"
NODES=(28.59.80.88 28.59.84.154 28.59.84.48 29.162.233.209)
MASTER=${NODES[0]}
PORT=29701
PY=/root/unirl/.venv/bin/python
NPROC=8
NNODES=4
LOGDIR=/root/shared/.clusters/.tmp
# NCCL env for this cluster's inter-node fabric (eth, no IB assumed; relax timeouts)
ENV="NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond1 NCCL_P2P_DISABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
for i in "${!NODES[@]}"; do
  N=${NODES[$i]}
  CMD="cd /root/shared/.clusters/.script && $ENV $PY -m torch.distributed.run \
--nnodes=$NNODES --node_rank=$i --nproc_per_node=$NPROC \
--master_addr=$MASTER --master_port=$PORT $SCRIPT $EXTRA \
> $LOGDIR/${TAG}_rank${i}.log 2>&1 &"
  bash "$CB" "$N" run "$CMD echo launched_node${i}_pid \$!" 2>&1 | grep -E 'launched_node|pid' | tail -1
done
echo "master=$MASTER tag=$TAG -> logs $LOGDIR/${TAG}_rank*.log"
