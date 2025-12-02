python train_topo.py \
    --experiment_name "topo_class_token" \
    --curve_patch_checkpoint experiments/dt_kl/ckpt/best_model.pth \
    --batch_size 2 \
    --num_workers 2 \
    --max_halfedges 100000 \
    --hidden_dim 384 \
    --num_layers 4 \
    # --checkpoint experiments/topo_class_token/ckpt/best_model.pth \
