import os
import cv2
import argparse
import warnings
import trimesh
import torch
import torch.utils
from tqdm import tqdm
from llava import conversation as conversation_lib
from PIL import Image
import numpy as np

from posegpt.utils import Config
from posegpt.utils.rotation_conversions import axis_angle_to_matrix

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from utils.load import load_cot_pose_model
from utils.vis import vis_mesh

def main(args):
    # disable_torch_init()
    config = Config.fromfile(args.config)
    torch_dtype = torch.bfloat16
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f'cuda:{local_rank}'
    conversation_lib.default_conversation = conversation_lib.conv_templates['mistral_instruct']

    # build model, tokenizer
    print('Load model...')
    model, image_processor = load_cot_pose_model(
        config, args.ft_cot_path, 'cache/unipose_merged', torch_dtype=torch_dtype, device_map={"": local_rank}, **config)

    while True:
        prompt = input('=> User: ')

        body_poseA_rotmat = torch.zeros((22, 3, 3))
        body_poseB_rotmat = torch.zeros((22, 3, 3))

        imageA = torch.zeros((3, 336, 336))
        imageB = torch.zeros((3, 336, 336))
        hmr_imageA = torch.zeros((3, 256, 256))
        hmr_imageB = torch.zeros((3, 256, 256))

        batch = dict(
            body_poseA_rotmat=body_poseA_rotmat.to(torch.bfloat16).to(device).unsqueeze(0),
            body_poseB_rotmat=body_poseB_rotmat.to(torch.bfloat16).to(device).unsqueeze(0),
            images=torch.stack([imageA, imageB], dim=0).to(torch.bfloat16).to(device),
            hmr_images=torch.stack([hmr_imageA, hmr_imageB], dim=0).to(torch.bfloat16).to(device),
            tasks=[{'input': prompt}],
            caption=['']
        )

        with torch.no_grad():
            output = model.evaluate(**batch)

        body_pose = output['body_pose']
        text = output['text']
        if text is not None:
            print(f'=> GPT: {text[0]}')
        if body_pose is not None:
            body_pose = body_pose.to(torch.float32).cpu().squeeze(0).numpy()
            np.save('smpl_mesh_rotmat.npy', axis_angle_to_matrix(torch.from_numpy(body_pose).view(-1, 3)))
            vis_mesh(body_pose)

            print("SMPL mesh saved as smpl_mesh.obj")
            print("SMPL parameters saved as smpl_mesh_rotmat.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_cot_path", type=str, required=True)
    parser.add_argument("--config", type=str, default='configs/inference.py')
    args = parser.parse_args()

    main(args)
