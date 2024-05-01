pip install git+https://github.com/tencent-ailab/IP-Adapter.git
pip install -r requirements.txt
# git submodule update --init

# cd OOTDiffusion
git clone -b dev https://github.com/camenduru/OOTDiffusion
apt -y install -qq aria2
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/images/demo.png -d OOTDiffusion/images -o demo.png
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/humanparsing/exp-schp-201908261155-lip.pth -d OOTDiffusion/checkpoints/humanparsing -o exp-schp-201908261155-lip.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/humanparsing/exp-schp-201908301523-atr.pth -d OOTDiffusion/checkpoints/humanparsing -o exp-schp-201908301523-atr.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/openpose/ckpts/body_pose_model.pth -d OOTDiffusion/checkpoints/openpose/ckpts -o body_pose_model.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/feature_extractor/preprocessor_config.json -d OOTDiffusion/checkpoints/ootd/feature_extractor -o preprocessor_config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm/config.json -d OOTDiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm -o config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm/diffusion_pytorch_model.safetensors -d OOTDiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm -o diffusion_pytorch_model.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton/config.json -d OOTDiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton -o config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton/diffusion_pytorch_model.safetensors -d OOTDiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton -o diffusion_pytorch_model.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/scheduler/scheduler_config.json -d OOTDiffusion/checkpoints/ootd/scheduler -o scheduler_config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/text_encoder/config.json -d OOTDiffusion/checkpoints/ootd/text_encoder -o config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/ootd/text_encoder/pytorch_model.bin -d OOTDiffusion/checkpoints/ootd/text_encoder -o pytorch_model.bin
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/tokenizer/merges.txt -d OOTDiffusion/checkpoints/ootd/tokenizer -o merges.txt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/tokenizer/special_tokens_map.json -d OOTDiffusion/checkpoints/ootd/tokenizer -o special_tokens_map.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/tokenizer/tokenizer_config.json -d OOTDiffusion/checkpoints/ootd/tokenizer -o tokenizer_config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/tokenizer/vocab.json -d OOTDiffusion/checkpoints/ootd/tokenizer -o vocab.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/vae/config.json -d OOTDiffusion/checkpoints/ootd/vae -o config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/resolve/main/checkpoints/ootd/vae/diffusion_pytorch_model.bin -d OOTDiffusion/checkpoints/ootd/vae -o diffusion_pytorch_model.bin
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OOTDiffusion/raw/main/checkpoints/ootd/model_index.json -d OOTDiffusion/checkpoints/ootd -o model_index.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/SUPIR/resolve/main/clip-vit-large-patch14.tar -d OOTDiffusion/checkpoints -o clip-vit-large-patch14.tar

mkdir -p OOTDiffusion/checkpoints/clip-vit-large-patch14
cd OOTDiffusion/checkpoints/clip-vit-large-patch14
tar -xvf ../clip-vit-large-patch14.tar 
cd ../../..
# tar -xvf OOTDiffusion/checkpoints/clip-vit-large-patch14.tar -C OOTDiffusion/checkpoints/clip-vit-large-patch14
