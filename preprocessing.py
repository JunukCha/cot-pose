import json
import numpy as np
import os
import tqdm
from collections import defaultdict
import pickle
import argparse
from datasets import Dataset, DatasetDict

import torch
from posegpt.models.posegpt import batch_pose_tokenids2string
from posegpt.utils import Config
from posegpt.utils.rotation_conversions import axis_angle_to_matrix
from utils.load import load_unipose_model

import constants

def preprocess_triplets(task='train'):
    # Load PoseScript
    with open(constants.posescript_auto_json_path, 'r') as f:
        posescript_data = json.load(f)

    # PoseScript → AMASS mapping
    with open(constants.posescript_amass_mapping_json_path, 'r') as f:
        posescript_to_amass = json.load(f)

    # Load BABEL
    with open(constants.babel_json_path.format(task=task), 'r') as f:
        babel_raw = json.load(f)

    # babel_data_by_feat: dict[str, list[dict]]
    babel_data_by_feat = defaultdict(list)
    for babel_id, info in babel_raw.items():
        feat_p = info['feat_p']
        feat_p = '/'.join(feat_p.split('/')[1:])
        babel_data_by_feat[feat_p].append(info)

    triplets = []

    no_npz_path = []
    for pose_id, descs in tqdm.tqdm(posescript_data.items()):
        if pose_id not in posescript_to_amass:
            print(f"[SKIP] pose_id {pose_id} not in posescript_to_amass")
            continue

        data_name, seq_name_poses, frame_idx = posescript_to_amass[pose_id]
        npz_path = f'data/AMASS/{seq_name_poses}'
        if not os.path.exists(npz_path):
            print(f"[SKIP] npz file not found: {npz_path}")
            no_npz_path.append(npz_path.split("/")[0])
            continue

        if seq_name_poses not in babel_data_by_feat:
            print(f"[SKIP] no babel entry for {seq_name_poses}")
            continue

        amass_data = np.load(npz_path)
        try:
            if 'root_orient' in amass_data and 'pose_body' in amass_data:
                poses_param = np.concatenate([
                    amass_data['root_orient'],
                    amass_data['pose_body']
                ])
            else:
                poses_param = amass_data['poses'][:, :66]
            num_frames = poses_param.shape[0]
            pose_param = poses_param[frame_idx]
        except Exception as e:
            print(f"[SKIP] error loading npz: {npz_path}, error: {e}")
            continue

        for babel_info in babel_data_by_feat[seq_name_poses]:
            duration = babel_info['dur']
            fps = num_frames / duration

            ann = babel_info['frame_ann']
            frame_ann = True
            if ann is None:
                ann = babel_info['seq_ann']
                frame_ann = False
            labels = ann['labels']

            found = False
            for label in labels:
                if frame_ann:
                    start_f = int(label['start_t'] * fps)
                    end_f = int(label['end_t'] * fps)
                    if start_f <= frame_idx <= end_f:
                        triplets.append({
                            'pose_id': pose_id,
                            'pose_param': pose_param.tolist(),
                            'action_class': label['proc_label'],
                            'description': descs,
                            'frame_idx': frame_idx,
                            'amass_seq': seq_name_poses
                        })
                        found = True
                else:
                    triplets.append({
                        'pose_id': pose_id,
                        'pose_param': pose_param.tolist(),
                        'action_class': label['proc_label'],
                        'description': descs,
                        'frame_idx': frame_idx,
                        'amass_seq': seq_name_poses
                    })
                    found = True
            if not found:
                print(f"[SKIP] no label match for pose_id {pose_id} in frame {frame_idx}")
    print(set(no_npz_path))
    print(f'# of matched triplets: {len(triplets)}')
    if triplets:
        print(triplets[0])

    with open(f'triplets_{task}.pkl', 'wb') as f:
        pickle.dump(triplets, f)

def save_hf_dataset():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default='cache/unipose')
    parser.add_argument("--model-base", type=str, default='cache/llava-v1.6-mistral-7b')
    parser.add_argument("--config", type=str, default='configs/inference.py')
    args = parser.parse_args()

    np.random.seed(0)

    config = Config.fromfile(args.config)
    torch_dtype = torch.bfloat16
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f'cuda:{local_rank}'

    model, image_processor = load_unipose_model(
        config, args.model_path, args.model_base, torch_dtype=torch_dtype, device_map={"": local_rank}, **config
    )
    with open(f'data/triplets_train.pkl', 'rb') as f:
        triplets_train = pickle.load(f)
    with open(f'data/triplets_train_syn_refined.pkl', 'rb') as f:
        triplets_train_syn = pickle.load(f)
    with open(f'data/triplets_val.pkl', 'rb') as f:
        triplets_val = pickle.load(f)
    with open(f'data/triplets_real_test_refined.pkl', 'rb') as f:
        triplets_test = pickle.load(f)

    train_list = []
    train_action_classes = []
    for triplet in tqdm.tqdm(triplets_train):
        pose_id = triplet['pose_id']
        action_class = triplet['action_class']
        if action_class not in train_action_classes:
            train_action_classes.append(action_class)
        else:
            continue
        descriptions = triplet['description']

        pose_param = torch.FloatTensor(triplet['pose_param']).reshape(-1, 3)
        pose_rotmat = axis_angle_to_matrix(pose_param)
        pose_rotmat = pose_rotmat.to(torch_dtype).to(device).unsqueeze(0)
        pose_ids = model.model.pose_vqvae.encode(pose_rotmat)
        pose_string = batch_pose_tokenids2string(pose_ids, model.pose_vqvae_codebook_size)[0]

        for description in descriptions[-1:]:
            # action = np.random.choice(templates).format(action=action_class)
            reasoning = description[:-1] if description[-1] == '.' else description
            answer = pose_string + '.'
            train_list.append({
                "pose_id": pose_id,
                "action": action_class,
                "reasoning": reasoning,
                "reasoning_refined": "",
                "answer": answer,
                "pose_param": triplet['pose_param'],
            })

    for triplet in tqdm.tqdm(triplets_train_syn):
        pose_id = 'syn'
        action_class = triplet['action_class']
        descriptions = triplet['description']
        descriptions_refined = triplet['description_gpt_4o_mini']

        pose_param = torch.FloatTensor(triplet['pose_param']).reshape(-1, 3)
        pose_rotmat = axis_angle_to_matrix(pose_param)
        pose_rotmat = pose_rotmat.to(torch_dtype).to(device).unsqueeze(0)
        pose_ids = model.model.pose_vqvae.encode(pose_rotmat)
        pose_string = batch_pose_tokenids2string(pose_ids, model.pose_vqvae_codebook_size)[0]

        for description in descriptions[-1:]:
            reasoning = description[:-1] if description[-1] == '.' else description
            reasoning_refined = descriptions_refined[:-1] if descriptions_refined[-1] == '.' else descriptions_refined
            answer = pose_string + '.'
            train_list.append({
                "pose_id": pose_id,
                "action": action_class,
                "reasoning": reasoning,
                "reasoning_refined": reasoning_refined,
                "answer": answer,
                "pose_param": triplet['pose_param'],
            })

    val_list = []
    val_action_classes = []
    for triplet in tqdm.tqdm(triplets_val):
        pose_id = triplet['pose_id']
        action_class = triplet['action_class']
        if action_class not in val_action_classes:
            val_action_classes.append(action_class)
        else:
            continue
        descriptions = triplet['description']

        pose_param = torch.FloatTensor(triplet['pose_param']).reshape(-1, 3)
        pose_rotmat = axis_angle_to_matrix(pose_param)
        pose_rotmat = pose_rotmat.to(torch_dtype).to(device).unsqueeze(0)
        pose_ids = model.model.pose_vqvae.encode(pose_rotmat)
        pose_string = batch_pose_tokenids2string(pose_ids, model.pose_vqvae_codebook_size)[0]

        for description in descriptions[-1:]:
            reasoning = description[:-1] if description[-1] == '.' else description
            answer = pose_string + "."
            val_list.append({
                "pose_id": pose_id,
                "action": action_class,
                "reasoning": reasoning,
                "reasoning_refined": "",
                "answer": answer,
                "pose_param": triplet['pose_param'],
            })

    test_list = []
    test_action_classes = []
    for triplet in tqdm.tqdm(triplets_test):
        action_class = triplet['action_class']
        if action_class not in test_action_classes:
            test_action_classes.append(action_class)
        else:
            continue
        descriptions = triplet['description']

        pose_param = torch.FloatTensor(triplet['pose_param']).reshape(-1, 3)
        pose_rotmat = axis_angle_to_matrix(pose_param)
        pose_rotmat = pose_rotmat.to(torch_dtype).to(device).unsqueeze(0)
        pose_ids = model.model.pose_vqvae.encode(pose_rotmat)
        pose_string = batch_pose_tokenids2string(pose_ids, model.pose_vqvae_codebook_size)[0]

        for description in descriptions[-1:]:
            # instruction = np.random.choice(templates).format(action=action_class)
            reasoning = description[:-1] if description[-1] == '.' else description
            reasoning_refined = descriptions_refined[:-1] if descriptions_refined[-1] == '.' else descriptions_refined
            answer = pose_string + "."
            test_list.append({
                "pose_id": 'syn_test',
                "action": action_class,
                "reasoning": reasoning,
                "reasoning_refined": reasoning_refined,
                "answer": answer,
                "pose_param": triplet['pose_param'],
            })

    train_ds = Dataset.from_list(train_list)
    val_ds  = Dataset.from_list(val_list)
    test_ds  = Dataset.from_list(test_list)

    dataset_dict = DatasetDict({
        "train": train_ds,
        "val": val_ds,
        "test":  test_ds
    })
    dataset_dict.push_to_hub(
        'cot-poses-data',
    )

if __name__=='__main__':
    preprocess_triplets(task='train')
    preprocess_triplets(task='val')
    preprocess_triplets(task='test')
    save_hf_dataset()