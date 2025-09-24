

echo myuser=`whoami`
nvidia-smi

pwd
source .venv/bin/activate
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID

# accelerate launch --num_processes $(( 4 * $COUNT_NODE )) --num_machines $COUNT_NODE --machine_rank ${THEID} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} scripts/train_custom_sft.py configs/hyper_lora_decontam_lol_tasks.yaml --model_dir=mistralai/Mistral-7B-Instruct-v0.2 --emb_model=Alibaba-NLP/gte-large-en-v1.5 --warmup_frac=0.2 --lr=2.5e-5  --n_points_per_task=1 --grad_accum_steps=1 --epochs=5000 --n_descs_per_ds=128 --n_train_ds=479 --exp_setup=hyper_lora --encoder_type=linear --l2_reg_generated_w=1e-3 --label_smoothing=0.1 --neftune_noise_alpha=5 --weight_decay=1e-2

# accelerate launch --num_processes 4 --num_machines 1 scripts/train_custom_sft.py     configs/hyper_lora_decontam_lol_tasks.yaml     --model_dir=google/gemma-2-2b-it     --emb_model=Alibaba-NLP/gte-large-en-v1.5     --warmup_frac=0.2 --lr=2.5e-5      --n_points_per_task=1 --grad_accum_steps=1     --epochs=5000 --n_descs_per_ds=128 --n_train_ds=479 --encoder_type=linear     --l2_reg_generated_w=1e-3 --label_smoothing=0.1     --neftune_noise_alpha=5 --weight_decay=1e-2     --hypernet_latent_size=512 --head_in_size=2048 --val_batch_size=32 --z_type=diag --exp_setup=hyper_lora

CONFIG_FILE="configs/hyper_lora_decontam_lol_tasks.yaml"

echo "Starting accelerate script..."

EXTRA_ARGS=""
if [ "$DATASET_TYPE" = "align_OHE" ]; then
    CONFIG_FILE="configs/hyper_lora_align_OHE.yaml"
elif [ "$DATASET_TYPE" = "align" ]; then
    CONFIG_FILE="configs/hyper_lora_align.yaml"
elif [ "$DATASET_TYPE" = "align_OHE_region" ]; then
    if [ -z "$DS_PER_BATCH" ] || [ "$DS_PER_BATCH" -gt 5 ]; then
        DS_PER_BATCH=5
    fi
    CONFIG_FILE="configs/hyper_lora_align_OHE_region.yaml"
elif [ "$DATASET_TYPE" = "align_region" ]; then
    if [ -z "$DS_PER_BATCH" ] || [ "$DS_PER_BATCH" -gt 5 ]; then
        DS_PER_BATCH=5
    fi
    CONFIG_FILE="configs/hyper_lora_align_region.yaml"
# elif [ "$DATASET_TYPE" = "task_OHE" ]; then
#     CONFIG_FILE="configs/hyper_lora_decontam_lol_tasks.yaml"
#     EXTRA_ARGS="--use_one_hot_task_emb=True --use_per_task_emb=True --use_hierarchical_sampler=True"
else
    CONFIG_FILE="configs/hyper_lora_decontam_lol_tasks.yaml"
fi

# --use_inp_as_desc=True
if [ "$EXP_NAME" = "hyper_decoder" ]; then
    EXP_NAME="hyper_lora"
    # USE_INP_AS_DESC=True
    # USE_PER_TASK_EMBED=False
    EXTRA_ARGS=" --use_inp_as_desc=True --use_per_task_emb=False"
fi

N_EPOCHS=2000
if [[ "$DATASET_TYPE" = *"align"* ]]; then
    N_EPOCHS=5000
    EXTRA_ARGS+=" --keep_only_best=True"
fi

case "$MODEL_NAME" in
    mistral)
        MODEL_DIR="mistralai/Mistral-7B-Instruct-v0.2"
        ;;
    gemma)
        MODEL_DIR="google/gemma-2-2b-it"
        ;;
    llama)
        MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
        ;;
    *)
        echo "Unknown MODEL_NAME: $MODEL_NAME"
        exit 1
        ;;
esac
if [[ "$EXP_NAME" == lora_* && "$DATASET_TYPE" = *"task"* ]]; then
    BENCHMARK_NAME="${EXP_NAME#*_}"
    EVAL_DS_INFO="{${BENCHMARK_NAME}: {descriptions: [PLACEHOLDER]}}"
    accelerate launch \
      --num_processes $(( 4 * $COUNT_NODE )) \
      --num_machines $COUNT_NODE \
      --machine_rank ${THEID} \
      --main_process_ip ${MASTER_ADDR} \
      --main_process_port ${MASTER_PORT} \
      scripts/train_custom_sft.py \
      "configs/lora_gsm8k.yaml" \
      --save_dir=${SAVE_DIR} \
      --run_name=${RUN_NAME} \
      --r=${MAT_RANK} \
      --exp_setup=oracle_lora \
      --train_ds_names="${BENCHMARK_NAME}" \
      --eval_ds_info="${EVAL_DS_INFO}"
    uv run python scripts/run_eval.py --model-dir ${MODEL_DIR} \
        --lora-dirs ${SAVE_DIR} \
        --tasks ${BENCHMARK_NAME} \
        --save-results
elif [[ "$EXP_NAME" == lora_* && "$DATASET_TYPE" = *"align"* ]]; then
    CULTURE_NAME="${EXP_NAME#*_}"
    EVAL_DS_INFO="{cul_${CULTURE_NAME}: {descriptions: [PLACEHOLDER]}}"
    accelerate launch \
      --num_processes $(( 4 * $COUNT_NODE )) \
      --num_machines $COUNT_NODE \
      --machine_rank ${THEID} \
      --main_process_ip ${MASTER_ADDR} \
      --main_process_port ${MASTER_PORT} \
      scripts/train_custom_sft.py \
      "configs/lora_Germany.yaml" \
      --save_dir=${SAVE_DIR} \
      --run_name=${RUN_NAME} \
      --r=${MAT_RANK} \
      --train_ds_names="cul_${CULTURE_NAME}" \
      --eval_ds_info="${EVAL_DS_INFO}"
elif [ "$EXP_NAME" = "mt_lora" ]; then
    accelerate launch \
    --num_processes $(( 4 * $COUNT_NODE )) \
    --num_machines $COUNT_NODE \
    --machine_rank ${THEID} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    scripts/train_custom_sft.py \
    ${CONFIG_FILE} \
    --model_dir=${MODEL_DIR} \
    --n_train_ds=479 \
    --lr=2.5e-5 \
    --warmup_frac=0.2 \
    --n_ds_per_batch=${DS_PER_BATCH} \
    --n_points_per_task=1 \
    --grad_accum_steps=1 \
    --epochs=${N_EPOCHS} \
    --exp_setup=${EXP_NAME} \
    --use_per_task_emb=False \
    --label_smoothing=0.1 \
    --weight_decay=1e-3 \
    --neftune_noise_alpha=5 \
    --val_batch_size=32 \
    --save_dir=${SAVE_DIR} \
    --run_name=${RUN_NAME} \
    --r=${MAT_RANK} \
    ${EXTRA_ARGS}
elif [ "$MODEL_NAME" = "mistral" ]; then
    accelerate launch \
    --num_processes $(( 4 * $COUNT_NODE )) \
    --num_machines $COUNT_NODE \
    --machine_rank ${THEID} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    scripts/train_custom_sft.py \
    ${CONFIG_FILE} \
    --model_dir=${MODEL_DIR} \
    --emb_model=Alibaba-NLP/gte-large-en-v1.5 \
    --warmup_frac=0.2 \
    --lr=2.5e-5 \
    --n_points_per_task=1 \
    --grad_accum_steps=1 \
    --epochs=${N_EPOCHS} \
    --n_descs_per_ds=128 \
    --n_train_ds=479 \
    --exp_setup="${EXP_NAME}" \
    --encoder_type=linear \
    --l2_reg_generated_w=1e-3 \
    --label_smoothing=0.1 \
    --neftune_noise_alpha=5 \
    --weight_decay=1e-2 \
    --save_dir=${SAVE_DIR} \
    --run_name=${RUN_NAME} \
    --r=${MAT_RANK} \
    --n_ds_per_batch=${DS_PER_BATCH} \
    --z_type=${Z_TYPE} \
    ${EXTRA_ARGS}
elif [ "$MODEL_NAME" = "gemma" ]; then
    accelerate launch \
    --num_processes $(( 4 * $COUNT_NODE )) \
    --num_machines $COUNT_NODE \
    --machine_rank ${THEID} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    scripts/train_custom_sft.py \
    ${CONFIG_FILE} \
    --model_dir=${MODEL_DIR} \
    --emb_model=Alibaba-NLP/gte-large-en-v1.5 \
    --warmup_frac=0.2 --lr=2.5e-5  \
    --n_points_per_task=1 --grad_accum_steps=1 \
    --epochs=${N_EPOCHS} --n_descs_per_ds=128 --n_train_ds=479 \
    --exp_setup="${EXP_NAME}" --encoder_type=linear \
    --l2_reg_generated_w=1e-3 --label_smoothing=0.1 \
    --neftune_noise_alpha=5 --weight_decay=1e-2 \
    --hypernet_latent_size=512 --head_in_size=2048 --val_batch_size=32 \
    --save_dir=${SAVE_DIR} \
    --run_name=${RUN_NAME} \
    --r=${MAT_RANK} \
    --n_ds_per_batch=${DS_PER_BATCH}  \
    --z_type=${Z_TYPE} \
    ${EXTRA_ARGS}
    # accelerate launch \
    # --num_processes $(( 4 * $COUNT_NODE )) \
    # --num_machines $COUNT_NODE \
    # --machine_rank ${THEID} \
    # --main_process_ip ${MASTER_ADDR} \
    # --main_process_port ${MASTER_PORT} \
    # scripts/train_custom_sft.py \
    # configs/hyper_lora_decontam_lol_tasks_fast.yaml \
    # --model_dir=google/gemma-2-2b-it \
    # --emb_model=Alibaba-NLP/gte-large-en-v1.5 \
    # --warmup_frac=0.2 --lr=2.5e-5  \
    # --n_points_per_task=1 --grad_accum_steps=1 \
    # --epochs=100 --n_descs_per_ds=128 --n_train_ds=4 --n_ds_per_batch=4 \
    # --exp_setup="${EXP_NAME}" --encoder_type=linear \
    # --l2_reg_generated_w=1e-3 --label_smoothing=0.1 \
    # --neftune_noise_alpha=5 --weight_decay=1e-2 \
    # --hypernet_latent_size=64 --head_in_size=256 --val_batch_size=32 \
    # --save_dir=${SAVE_DIR} \
    # --run_name=${RUN_NAME} \
    # --r=${MAT_RANK}
elif [ "$MODEL_NAME" = "llama" ]; then
    accelerate launch \
    --num_processes $(( 4 * $COUNT_NODE )) \
    --num_machines $COUNT_NODE \
    --machine_rank ${THEID} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    scripts/train_custom_sft.py \
    ${CONFIG_FILE} \
    --model_dir=${MODEL_DIR} \
    --emb_model=Alibaba-NLP/gte-large-en-v1.5 \
    --warmup_frac=0.2 --lr=2.5e-5  \
    --n_points_per_task=1 --grad_accum_steps=1 \
    --epochs=${N_EPOCHS} --n_descs_per_ds=128 --n_train_ds=479 \
    --exp_setup="${EXP_NAME}" --encoder_type=linear \
    --l2_reg_generated_w=1e-3 --label_smoothing=0.1 \
    --neftune_noise_alpha=5 --weight_decay=1e-2 \
    --hypernet_latent_size=512 --head_in_size=2048 \
    --val_batch_size=32 \
    --save_dir=${SAVE_DIR} \
    --run_name=${RUN_NAME} \
    --r=${MAT_RANK} \
    --n_ds_per_batch=${DS_PER_BATCH} \
    --z_type=${Z_TYPE} \
    ${EXTRA_ARGS}
else
    echo "Unknown MODEL_NAME: $MODEL_NAME"
    exit 1
fi