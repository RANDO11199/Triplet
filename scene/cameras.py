#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.

## Copyright (C) 2024, Jiajie Yang https://github.com/RANDO11199
# All rights reserved.

# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from pytorch3d.renderer.cameras import CamerasBase,PerspectiveCameras,FoVPerspectiveCameras
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.dataset_readers import CameraInfo,parse_camera_model
import math
# CV Style
class Camera(nn.Module):
    def __init__(self, 
                 cam_info:CameraInfo,
                 image:torch.Tensor,uid,
                 trans=np.array([0.0, 0.0, 0.0]),
                 scale=1.0, gt_alpha_mask:torch.Tensor = None, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = cam_info.uid
        self.image_name = cam_info.image_name
        
        R = torch.from_numpy(cam_info.R)
        T = torch.from_numpy(cam_info.T)
        # R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1) # RDF -> LUF

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None
        intrinsics, correction,model_type = parse_camera_model(cam_model=cam_info.camera_model)
        intrinsics = intrinsics / scale

        if intrinsics.shape[0] == 6:
            self.fcl_screen = ((intrinsics[0], intrinsics[1]),)
            self.prp_screen = ((intrinsics[2], intrinsics[3]),)
            fovx = intrinsics[4]
            fovy = intrinsics[5]
            aspect_ratio = fovx/fovy
        elif intrinsics.shape[0] == 5:
            self.fcl_screen = ((intrinsics[0],intrinsics[0]),)
            self.prp_screen = ((intrinsics[1], intrinsics[2]),)
            fovx = intrinsics[3]
            fovy = intrinsics[4]
            aspect_ratio = fovx/fovy
        if correction is not None:
            # TODO: Support more camera 
            raise NotImplementedError(f'Only pinhole or simple pinhole is supported now.')
        else:
            # self.camera = FoVPerspectiveCameras(
            #     znear=0.01,
            #     zfar=100,
            #     fov=(fovy/math.pi)*180,
            #     aspect_ratio= aspect_ratio,
            #     # in_ndc=False,
            #     R = R[None],
            #     T = T[None],
            #     device=self.data_device
            # )
            self.camera = PerspectiveCameras(
                focal_length=self.fcl_screen,
                principal_point= self.prp_screen,
                in_ndc=False,
                image_size= ((self.image_height,self.image_width),),
                R = R[None],
                T = T[None],
                device=self.data_device
            )

        self.scale = scale
        self.world_view_transform = self.camera.get_world_to_view_transform()
        self.projection_matrix = self.camera.get_projection_transform()
        self.full_proj_transform = self.camera.get_full_projection_transform()
        self.camera_center = self.camera.get_camera_center()
    
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def setup_camera(camera, camera_type = None,downsample_rate = 1):
    # Initialize a camera.
    if camera is not None:
        c2w = torch.inverse(camera['w2c'])
        R, T = c2w[:3, :3], c2w[:3, 3:]

        R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1) # from RDF to Left-Up-Forward for Rotation

        new_c2w = torch.cat([R, T], 1)
        w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]])), 0))
        R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3]
        R = R[None] # batch 1 for rendering
        T = T[None]

        H, W = camera['H'], camera['W']
        H, W = int(H / downsample_rate), int(W /downsample_rate)

        intrinsics = camera['intrinsics'] / downsample_rate

        image_size = ((H, W),)  # (h, w)
        fcl_screen = (intrinsics[0],)  # Fix Error in Original Code fcl_screen = ((intrinsics[0], intrinsics[1]),) =>fcl_screen = ((intrinsics[0],),)
        prp_screen = ((intrinsics[1], intrinsics[2]), )  # Fix Error in Original Code prp_screen = ((intrinsics[2], intrinsics[3]), ) => prp_screen = ((intrinsics[1], intrinsics[2]), ) 
        cameras = PerspectiveCameras(focal_length=fcl_screen, principal_point=prp_screen, in_ndc=False, image_size=image_size, R=R, T=T, device=device)

    elif camera_type == 'fovperspective':
        R, T = look_at_view_transform(2.7, 0, 180) 
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        image_size = ((255),) # args.image_size
    else:
        raise Exception("Undefined camera")
    print('Camera R T', R, T)

    # Define the settings for rasterization and shading.
    print(image_size)
    return cameras,H,W


def CoordniateSystemConversion(w2c: torch.Tensor ,return_RT: bool =True) -> torch.Tensor:
    c2w = torch.inverse(w2c)
    R, T = c2w[:3, :3], c2w[:3, 3:]
    R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1) # from RDF to Left-Up-Forward for Rotation
    new_c2w = torch.cat([R, T], 1)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]])), 0))
    if return_RT:
        R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3]
        R = R[None] # batch 1 for rendering
        T = T[None]

        return R,T
    else:
        return w2c