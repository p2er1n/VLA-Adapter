vlm_path=/root/autodl-tmp/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b
data_root_dir=/root/autodl-tmp/data/libero
data_name=realworld
run_root_dir=/root/autodl-tmp/outputs
logs_dir=/root/autodl-tmp/logs
wandb_dir=/root/autodl-tmp/VLA-Adapter/wandb
run_note=
log_file="$logs_dir"/VLA-Adapter-realworld--"$data_name"--"$run_note"--"$bs-$grad_acc_steps-$lr"--$current_time.log

bs=8
grad_acc_steps=2
lr=2e-4
shuffle_buffer_size=10000

# Start wandb sync background process
(
    while true; do
        if [ -d "$wandb_dir"/latest-run ]; then
            wandb sync "$wandb_dir"/latest-run 2>/dev/null
        fi
        sleep 60
    done
) &
wandb_sync_pid=$!

# Cleanup function to kill wandb sync process when script exits
cleanup() {
    kill $wandb_sync_pid 2>/dev/null
}
trap cleanup EXIT

current_time=$(date +"%Y-%m-%d-%H-%M-%S")
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path "$vlm_path" \
--config_file_path pretrained_models/configs \
--data_root_dir "$data_root_dir" \
--dataset_name $data_name \
--run_root_dir "$run_root_dir" \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--shuffle_buffer_size "$shuffle_buffer_size" \
--num_steps_before_decay 200000 \
--max_steps 200005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size "$bs" \
--grad_accumulation_steps "$grad_acc_steps" \
--learning_rate "$lr" \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "sjh-xidian-university" \
--wandb_project "VLA-Adapter-realworld" \
--run_id_note VLA-Adapter-realworld--"$data_name"--"$run_note"--"$bs-$grad_acc_steps-$lr-$shuffle_buffer_size"--$current_time \
2>&1 | tee "$log_file"
