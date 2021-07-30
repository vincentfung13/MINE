#!/bin/bash
cd $(dirname $0)

# distributed setting
N_NODES="1"
NODE_RANK="0"
MASTER_ADDR="localhost"
MASTER_PORT="8888"
GPUS_PER_NODE="1"

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            # Params for torch.distributed.launch
            MASTER_ADDR)              MASTER_ADDR=${VALUE} ;;
            MASTER_PORT)              MASTER_PORT=${VALUE} ;;
            N_NODES)                  N_NODES=${VALUE} ;;
            GPUS_PER_NODE)            GPUS_PER_NODE=${VALUE} ;;
            NODE_RANK)                NODE_RANK=${VALUE} ;;

            # Params for train.py
            WORKSPACE)                WORKSPACE=${VALUE} ;;
            DATASET)                  DATASET=${VALUE} ;;
            VERSION)                  VERSION=${VALUE} ;;
            EXTRA_CONFIG)             EXTRA_CONFIG=${VALUE} ;;
            *)
    esac
done

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Number of nodes: $N_NODES"
echo "Number of GPUs per node: $GPUS_PER_NODE"
echo "Node rank: $NODE_RANK"

# training settings
echo "Workspace: $WORKSPACE"
echo "Dataset: $DATASET"
echo "Version: $VERSION"
echo "Extra config: $EXTRA_CONFIG"

# If training with llff, pull the dataset from HDFS to local disk
LLFF="llff"
FLOWERS="flowers"
KITTIRAW='kitti_raw'
DTU="dtu"
if [ "$DATASET" = "$LLFF" ];
then
    DEFAULT_PARAMS="./configs/params_llff.yaml"
elif [ "$DATASET" = "$FLOWERS"  ];
then
    DEFAULT_PARAMS="./configs/params_flowers.yaml"
elif [ "$DATASET" = "$KITTIRAW" ];
then
    DEFAULT_PARAMS="./configs/params_kitti_raw.yaml"
elif [ "$DATASET" = "$DTU" ];
then
    DEFAULT_PARAMS="./configs/params_dtu.yaml"
else
    DEFAULT_PARAMS="./configs/params_realestate.yaml"
fi
echo "default params: $DEFAULT_PARAMS"

exec python3 -m torch.distributed.launch \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --nnodes $N_NODES \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank $NODE_RANK train.py \
    --config_path $DEFAULT_PARAMS \
    --workspace $WORKSPACE --version $VERSION \
    --extra_config "$EXTRA_CONFIG"
