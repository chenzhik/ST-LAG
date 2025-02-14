# LR video generation
CUDA_VISIBLE_DEVICES=0 python -m scripts.animate --exp_config configs/ablation/inference.yaml --H 2048 --W 2048 --L 32 --xformers

# HR video generation
CUDA_VISIBLE_DEVICES=0 python -m scripts.animate_st_grouping --exp_config configs/ablation/inference.2k32f.yaml --H 2048 --W 2048 --L 32 --xformers
