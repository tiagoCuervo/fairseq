#!/bin/bash
seed=$1

export FAIRSEQ_ROOT='/opt/fairseq'
export RVAD_ROOT='/opt/rVADfast'
export KENLM_ROOT='/opt/kenlm/build/bin'
export KALDI_ROOT='/opt/pykaldi/tools/kaldi'
export PATH=$PATH:/opt/conda/bin
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

PREFIX=wav2itl_exp

CONFIG_NAME=wav2itl
TASK_DATA=~/data/clarity_CPC2_data_16k/clarity_data/HA_outputs/train.2/feats_hubert_l/
LAST_DIR=$(basename "$TASK_DATA")
MODEL_NAME=${LAST_DIR#feats_}

if [[ $MODEL_NAME == "whisper" ]]; then
    IN_DIM=1280
elif [[ $MODEL_NAME == "hubert_xl" ]]; then
    IN_DIM=1280
else
    IN_DIM=1024
fi

NOW=`date '+%F_%H:%M:%S'`

PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
    -m --config-dir $FAIRSEQ_ROOT/examples/wav2itl/config \
    --config-name $CONFIG_NAME \
    task.data=${TASK_DATA} \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2itl \
    checkpoint.save_dir=cpc2_${MODEL_NAME}_tl_is23_${NOW}_${seed} \
    model.in_dim=${IN_DIM} common.seed=${seed} model.avg_pool=false

TASK_DATA=~/data/clarity_CPC2_data_16k/clarity_data/HA_outputs/train.3/feats_hubert_l/

NOW=`date '+%F_%H:%M:%S'`

PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
    -m --config-dir $FAIRSEQ_ROOT/examples/wav2itl/config \
    --config-name $CONFIG_NAME \
    task.data=${TASK_DATA} \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2itl \
    checkpoint.save_dir=cpc2_${MODEL_NAME}_tl_is23_${NOW}_${seed} \
    model.in_dim=${IN_DIM} common.seed=${seed} model.avg_pool=false

# TASK_DATA=~/data/clarity_CPC2_data_16k/clarity_data/HA_outputs/train.3/feats_wavlm/

# NOW=`date '+%F_%H:%M:%S'`

# PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
#     -m --config-dir $FAIRSEQ_ROOT/examples/wav2itl/config \
#     --config-name $CONFIG_NAME \
#     task.data=${TASK_DATA} \
#     common.user_dir=${FAIRSEQ_ROOT}/examples/wav2itl \
#     checkpoint.save_dir=cpc2_${MODEL_NAME}_tl_is23_${NOW}_${seed} \
#     model.in_dim=${IN_DIM} common.seed=${seed} model.avg_pool=false



    # checkpoint.save_dir=/multirun/2023-07-19/06-12-48/0/cpc2_wavlm_tl_is23_2023-07-19_06:12:40_1 \
    # checkpoint.save_dir=cpc2_${MODEL_NAME}_tl_is23_${NOW}_${seed} \
    # checkpoint.save_dir=/multirun/2023-07-19/07-40-04/0/cpc2_wavlm_tl_is23_2023-07-19_07:39:53_2 \
    # checkpoint.save_dir=/multirun/2023-07-19/07-48-47/0/cpc2_wavlm_tl_is23_2023-07-19_07:48:38_3 \
    # checkpoint.save_dir=/multirun/2023-07-19/07-12-03/0/cpc2_wavlm_tl_is23_2023-07-19_07:11:51_1 \
    # checkpoint.save_dir=/multirun/2023-07-19/09-03-02/0/cpc2_wavlm_tl_is23_2023-07-19_09:02:53_2 \
    # checkpoint.save_dir=/multirun/2023-07-19/11-48-01/0/cpc2_wavlm_tl_is23_2023-07-19_11:46:58_3 \
    # checkpoint.save_dir=/multirun/2023-07-19/01-19-54/0/cpc2_whisper_tl_is23_2023-07-19_01:18:57_2
    # checkpoint.save_dir=/multirun/2023-07-19/11-48-01/0/cpc2_wavlm_tl_is23_2023-07-19_11:46:58_3