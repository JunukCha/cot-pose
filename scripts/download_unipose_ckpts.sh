mkdir -p cache/
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir cache/clip-vit-large-patch14-336
huggingface-cli download liuhaotian/llava-v1.6-mistral-7b --local-dir cache/llava-v1.6-mistral-7b
huggingface-cli download L-yiheng/UniPose --local-dir cache/unipose
huggingface-cli download Jucha/unipose_merged --local-dir cache/unipose_merged

gdown https://drive.google.com/uc?id=1RZfB3oD2LitzQTKhc7dDX4IB0bY8yCEt -O cache/tokenhmr_model.ckpt
mkdir -p cache/pose_vqvae
gdown https://drive.google.com/uc?id=1S0khLj-45asaGFqAYYrw_iQ0hniqjL5j -O cache/pose_vqvae/best_MPJPE.ckpt