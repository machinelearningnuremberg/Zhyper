for model in "meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2" "google/gemma-2-2b-it";
do
    uv run python scripts/train_custom_sft.py \
        configs/hyper_lora_decontam_lol_tasks.yaml \
        --n_train_ds=479 \
        --model_dir=$model \
        --lr=2.5e-5 \
        --warmup_frac=0.2 \
        --n_ds_per_batch=8 \
        --n_points_per_task=1 \
        --grad_accum_steps=1 \
        --epochs=20000 \
        --exp_setup=mt_lora \
        --use_per_task_emb=False \
        --label_smoothing=0.1 \
        --weight_decay=1e-3 \
        --neftune_noise_alpha=5 \
        --val_batch_size=32
done