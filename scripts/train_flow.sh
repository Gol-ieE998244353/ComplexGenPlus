CUDA_VISIBLE_DEVICES=0,1,2 python train_flow.py \
    --experiment_name win \
    --batch_size 128 \
    --latent_folder ./latents \
    --vae_checkpoint experiments/aa/ckpt/checkpoint_epoch_1700.pth \
    --checkpoint experiments/win_flow_v13/ckpt/epoch_600.pth \
    --reset_scheduler \
    --logit_mean 0.0 \
    --logit_std 1.0
