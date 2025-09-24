WANDB_MODE=disabled uv run python scripts/train_custom_sft.py \
    configs/hyper_lora_decontam_lol_tasks.yaml \
    --model_dir=mistralai/Mistral-7B-Instruct-v0.2 \
    --emb_model=Alibaba-NLP/gte-large-en-v1.5 \
    --warmup_frac=0.2 --lr=2.5e-5 --n_ds_per_batch=8 \
    --n_points_per_task=1 --grad_accum_steps=1 \
    --epochs=2000 --n_descs_per_ds=128 --n_train_ds=479 \
    --exp_setup=hyper_lora --encoder_type=linear \
    --l2_reg_generated_w=1e-3 --label_smoothing=0.1 \
    --neftune_noise_alpha=5 --weight_decay=1e-2