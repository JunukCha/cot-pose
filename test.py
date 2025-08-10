import os, os.path as osp
import argparse
import torch
from llava import conversation as conversation_lib
from PIL import Image
import numpy as np
import torch.nn as nn

from posegpt.utils import Config
from posegpt.utils import BodyModel
from posegpt.utils.rotation_conversions import axis_angle_to_matrix

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from datasets import load_dataset
from scripts.instructions.finetune_instructions import action_template

from utils.load import (
    setup_models,
    load_model_retrieval,
    load_cot_pose_model,
)
from utils.vis import vis_mesh


def get_joints(pose=None):
    pose = torch.tensor(pose).to(torch.float64)
    pose_body = pose[None, 3:66]
    root_orient = pose[None, :3]
    smpl = BodyModel('cache/smpl_models/smplx/SMPLX_NEUTRAL.npz', dtype=torch.float64)
    p1 = smpl.forward(pose_body=pose_body, root_orient=root_orient)
    return p1.Jtr

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
        config, args.cot_pose_path, 'cache/unipose_merged', torch_dtype=torch_dtype, device_map={"": local_rank}, **config)

    models, _ = setup_models(['pretrained_models/retrieval/ret_distilbert_dataPSA2ftPSH2/seed1'], 'best', load_model_retrieval)
    posescript_retriever = models[0]

    cos = nn.CosineSimilarity()

    dataset = load_dataset("Jucha/cot-poses-data", split="test")

    pose_fids = 0
    text_fids = 0
    mm_fids = 0
    mpjpes = 0
    cnt = 0

    for item in dataset:
        action = item['action']
        reasoning = item['reasoning_refined']
        pose_param_gt = item['pose_param']
        pose_param_gt = torch.tensor(pose_param_gt).cuda()
        save_folder = osp.join('results', args.cot_pose_path, action)
        os.makedirs(save_folder, exist_ok=True)
        prompt = action_template.replace('<action>', action)

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

        pose_param_out = output['body_pose']
        text = output['text']
        if text is not None:
            print(f'=> GPT: {text[0]}')

        if pose_param_out is not None:
            # Copy GT and prediction to SMPL full pose format
            pose_data_gt = torch.zeros((1, 52, 3), device=pose_param_gt.device)
            pose_data_out = torch.zeros((1, 52, 3), device=pose_param_out.device)

            # Use GT global rotation for both GT and output
            pose_data_gt[:, :22] = pose_param_gt.reshape(1, 22, 3)
            pose_data_out[:, 0] = pose_param_gt.reshape(1, 22, 3)[:, 0]  # global rotation from GT
            pose_data_out[:, 1:22] = pose_param_out.reshape(1, 22, 3)[:, 1:22]

            # Compute pose features
            pose_feats_gt = posescript_retriever.encode_pose(pose_data_gt)
            pose_feats_out = posescript_retriever.encode_pose(pose_data_out)

            # Compute text features
            text_feats_gt = posescript_retriever.encode_raw_text(reasoning)
            text_feats_out = posescript_retriever.encode_raw_text(text[0].split('<pose')[0][:-2])

            # Compute metrics
            pose_fids += torch.sqrt(((pose_feats_gt - pose_feats_out) ** 2).mean())
            text_fids += torch.sqrt(((text_feats_gt - text_feats_out) ** 2).mean())
            mm_fids += torch.sqrt(((text_feats_out - pose_feats_out) ** 2).mean())

            # Save predicted mesh
            pose_param_out_np = pose_data_out[:, :22].to(torch.float32).squeeze(0).cpu().numpy().reshape(-1)
            np.save(
                osp.join(save_folder, 'smpl_mesh_rotmat.npy'),
                axis_angle_to_matrix(torch.from_numpy(pose_param_out_np).view(-1, 3))
            )

            # Visualize predicted and GT mesh
            pred_pose_vis = pose_param_out_np.copy()
            gt_pose_vis = pose_data_gt[:, :22].to(torch.float32).squeeze(0).cpu().numpy().reshape(-1)
            vis_mesh(
                pred_pose_vis, # 66 dim
                save_path=osp.join(save_folder, 'mesh_out.obj'),
                save_jpg_path=osp.join(save_folder, 'mesh_out.jpg')
            )
            vis_mesh(
                gt_pose_vis, # 66 dim
                save_path=osp.join(save_folder, 'mesh_gt.obj'),
                save_jpg_path=osp.join(save_folder, 'mesh_gt.jpg')
            )

            # MPJPE calculation
            joints_out = get_joints(pred_pose_vis)[:, :22]
            joints_gt = get_joints(gt_pose_vis)[:, :22]
            mpjpes += torch.norm(joints_out - joints_gt, dim=-1).mean()

            cnt += 1


    metrics = {
        'pose fid mean': pose_fids,
        'text fid mean': text_fids,
        'mm fid mean': mm_fids,
        'mpjpe mean': mpjpes,
    }

    with open(osp.join('results', args.cot_pose_path, 'metric.txt'), 'w') as f:
        for name, value in metrics.items():
            norm = (value / cnt).item()
            if "mpjpe" in name:
                norm *= 1000
                f.write(f"{name} {norm:.2f}\n")
                print(name, norm)
            elif "pose fid" in name:
                norm *= 1000
                f.write(f"{name} {norm:.4f}\n")
                print(name, norm)
            elif "text fid" in name or "mm fid" in name:
                norm *= 10
                f.write(f"{name} {norm:.4f}\n")
                print(name, norm)
            else:
                f.write(f"{name} {norm:.4f}\n")
                print(name, norm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default='cache/unipose')
    parser.add_argument("--model-base", type=str, default='cache/llava-v1.6-mistral-7b')
    parser.add_argument("--cot-pose-path", type=str, default='cache/cot-pose/full')
    parser.add_argument("--config", type=str, default='configs/inference.py')
    args = parser.parse_args()

    main(args)
