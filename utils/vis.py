import torch
import trimesh
from posegpt.utils import BodyModel


def vis_mesh(pose=None, pose_body=None, global_orient=None, save_path='smpl_mesh.obj'):
    if pose is not None:
        pose = torch.tensor(pose).to(torch.float64)
        pose_body = pose[None, 3:66]
        root_orient = pose[None, :3]
    else:
        pose_body = pose_body
        root_orient = global_orient
    smpl = BodyModel('cache/smpl_models/smplx/SMPLX_NEUTRAL.npz', dtype=torch.float64)
    p1 = smpl.forward(pose_body=pose_body, root_orient=root_orient)
    trimesh.Trimesh(vertices=p1.v.detach().numpy()[0], faces=smpl.f).export(save_path)
