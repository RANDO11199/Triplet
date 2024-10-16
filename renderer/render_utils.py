# Copyright (C) 2024, Jiajie Yang https://github.com/RANDO11199
# All rights reserved.
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
import pytorch3d
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorch3d.renderer import BlinnPhongShader,CookTorranceShader
# class Sobel(nn.Module):
#     def __init__(self):
#         super.__init__()
#         self.kernel_x = 
# It is not correct, but approximately good, will improve later
# should be done in camera space, but the answer is same
# TODO: see how py3d define NDC, why not in [-1,1]
def Init_shaer(opt):
    if opt.renderer == 'Rasterization':
        if opt.materials_type =="CT":
            return CookTorranceShader(device='cuda')
        elif opt.materials_type == 'BP':
            return BlinnPhongShader(device='cuda')
        else:
            raise NotImplementedError(f'Materials type: {opt.materials_type} for rasteriztion not implemented yet')
    else:
        raise NotImplementedError(f'renderer type: {opt.renderer} not implemented yet')
def get_normal_from_depth(z_buf):

    # Calculate the partial derivatives of depth with respect to x and y
    dx = cv2.Sobel(z_buf.cpu().numpy(), cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(z_buf.cpu().numpy(), cv2.CV_32F, 0, 1)

    # Compute the normal vector for each pixel
    normal = np.dstack((-dx, -dy, np.ones((800, 800))))
    norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

    # Map the normal vectors to the [0, 255] range and convert to uint8
    normal = (normal + 1) * 127.5
    normal = normal.clip(0, 255).astype(np.uint8)

    # Save the normal map to a file
    normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(output_path, normal_bgr)


def get_normals_from_fragments(meshes, fragments):
    """ https://github.com/facebookresearch/pytorch3d/issues/865 """
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]
    ones = torch.ones_like(fragments.bary_coords)
    pixel_normals = pytorch3d.ops.interpolate_face_attributes(
        fragments.pix_to_face, ones, faces_normals
    )
    return pixel_normals

def np_depth_to_colormap(depth,min_depth=-1.,max_depth = 10.):
    """ depth: [H, W] """
    depth_normalized = np.zeros(depth.shape)

    valid_mask = depth > -0.1 # valid
    # max_depth = np.max(depth)
    # depth[depth>max_depth] = max_depth

    # depth[depth<min_depth] = torch.tensor(max_depth)
    if valid_mask.sum() > 0:
        vmax = depth[valid_mask].max()
        vmin = depth[valid_mask].min()
        d_valid = depth[valid_mask]
        depth_normalized[valid_mask] = (d_valid - vmin) / (vmax - vmin)

        depth_np = (depth_normalized * 255).astype(np.uint8)
        # depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET) # Visually not good enough
        colormap = plt.get_cmap('viridis')
        depth_colormap = colormap(depth_np)
        # Convert from RGBA to RGB and scale to 8-bit [0, 255]
        depth_colormap = (depth_colormap[:, :, :3] )

        # depth_normalized = depth_normalized
    else:
        print('!!!! No depth projected !!!')
        depth_colormap = depth_normalized = np.zeros(depth.shape, dtype=np.uint8)
    return depth_colormap
# I write it myself? or fork from some place???
# Fix mismatch of coordinate system; Z-buffer is in ndc space. the x,y meshgrid is in screen space and need to align to zbuffer
def get_pseudo_normal_from_depth(fragment,target_cameras,pixel_alpha):
    
    # z_buffer = (fragment.zbuf[0,...] * pixel_alpha[0,...,0]).sum(-1) # TODO: Fix divergence
    z_buffer =( fragment.zbuf[0,...,0] -1)
    
    # z_buffer[z_buffer==-1] = 100. # -1
    x, y = torch.meshgrid(torch.arange(z_buffer.shape[0],dtype=torch.float,device='cuda'), torch.arange(z_buffer.shape[1],dtype=torch.float,device='cuda'))
    screen_point = torch.stack([x,y,torch.ones_like(y,dtype=y.dtype)],dim=-1)
    ndc_point = target_cameras.camera.get_screen_to_ndc(image_size=(target_cameras.image_height,target_cameras.image_width)).transform_points(screen_point)
    ndc_point[...,2] =z_buffer
    ndc_point = ndc_point.contiguous()
    world_normal = target_cameras.camera.unproject_points(ndc_point)
    # world_normal = torch.stack([((ndc_point[...,0] - target_cameras.prp_screen[0])/target_cameras.fcl_screen[0] * ndc_point[...,2]),
    #                             ((ndc_point[...,1] - target_cameras.prp_screen[1])/target_cameras.fcl_screen[1] * ndc_point[...,2]),
    #                              ndc_point[...,2]],dim = 0)
    world_normal = world_normal[None].permute(0,3,1,2)
    # kernel_x = torch.FloatTensor([[-0.125, 0, 0.125],
    #                                     [-0.25,0,0.25],
    #                                     [-0.125,0,0.125]]).view(1,1,3,3).repeat(1,1,1,1).cuda()
    # kernel_y = torch.FloatTensor([[-0.125,-0.25,-0.125],
    #                                     [0.,0,0.],
    #                                     [0.125,0.25,0.125]]).view(1,1,3,3).repeat(1,1,1,1).cuda()
    # kernel_x = torch.FloatTensor([[-0.0, 0, 0.0],
    #                                     [-1.0,0,1.0],
    #                                     [-0.0,0,0.0]]).view(1,1,3,3).repeat(1,1,1,1).cuda()
    # kernel_y = torch.FloatTensor([      [-0.0,-1.,-0.0],
    #                                     [0.,0,0.],
    #                                     [0.0,1.0,0.0]]).view(1,1,3,3).repeat(1,1,1,1).cuda()
    #R
    kernel_x = torch.FloatTensor([      [-3.0, 0, 3.0],
                                        [-10.0,0, 10.0],
                                        [-3.0, 0, 3.0]]).view(1,1,3,3).repeat(1,1,1,1).cuda()
    #D
    kernel_y = torch.FloatTensor([      [-3.0,-10.,-3.0],
                                         [0.,0, 0.],
                                        [3.0,10.0,3.0]]).view(1,1,3,3).repeat(1,1,1,1).cuda()
    dzdx = torch.stack([F.conv2d(world_normal[0,0][None],kernel_x,padding=1),
                        F.conv2d(world_normal[0,1][None],kernel_x,padding=1),
                        F.conv2d(world_normal[0,2][None],kernel_x,padding=1)],dim=1)
    dzdy = torch.stack([F.conv2d(world_normal[0,0][None],kernel_y,padding=1),
                        F.conv2d(world_normal[0,1][None],kernel_y,padding=1),
                        F.conv2d(world_normal[0,2][None],kernel_y,padding=1)],dim=1)
    #F point out screen
    # normal_pseudo = torch.stack([dzdx[0,1]*dzdy[0,2] - dzdx[0,2] * dzdy[0,1],
    #                     -dzdx[0,0]*dzdy[0,2] + dzdx[0,2]*dzdy[0,0],
    #                     dzdx[0,0]*dzdy[0,1]- dzdx[0,1]* dzdy[0,0]])
    # LUB -> LUF 
    # nomarl_pseudo[0] = nomarl_pseudo[0]
    # nomarl_pseudo[1] = nomarl_pseudo[1]
    # nomarl_pseudo[2] = nomarl_pseudo[2]
    # nomarl_pseudo = normal_pseudo/normal_pseudo.norm(dim=0,p=2)
    # nomarl_pseudo = torch.cross(dzdx,dzdy,dim=1)
    nomarl_pseudo = torch.cross(dzdx,dzdy,dim=1)
    # F Point out the screen
    normal_pseudo = F.normalize(nomarl_pseudo,dim=1)[0]

    # view_matrix = target_cameras.camera.get_world_to_view_transform().get_matrix()
    # nomarl_pseudo =(view_matrix[0][:3,:3] @ normal_pseudo.reshape(3,-1)).reshape(3,z_buffer.shape[0],z_buffer.shape[1])
    return normal_pseudo * (z_buffer!=-1).to(dtype=torch.float)
