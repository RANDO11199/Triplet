# Copyright (C) 2024, Jiajie Yang https://github.com/RANDO11199
# All rights reserved.
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
from pytorch3d.renderer import (RasterizationSettings,
                                MeshRasterizer,
                                HardPhongShader,)
# from pytorch3d.renderer.opengl import MeshRasterizerOpenGL
from scene.Triplet import TripletModel
import torch
from pytorch3d.ops import interpolate_face_attributes
from .render_utils import *
from .SH import *
       

def get_alpha_blend_pixel(alpha,faces,fragment):
    face_attribute_alpha = alpha[faces].unsqueeze(-1)
    pixel_alphas = interpolate_face_attributes(fragment.pix_to_face,
                                                    fragment.bary_coords,
                                                    face_attribute_alpha)
    precumprod = torch.cumprod(1- torch.cat([torch.zeros([pixel_alphas.shape[0],
                                                            pixel_alphas.shape[1],
                                                            pixel_alphas.shape[2],1,1],
                                                            device=pixel_alphas.device),
                                                            pixel_alphas],dim=3) ,dim=3)
    precomp_alpha =  pixel_alphas * precumprod[:,:,:,:-1,:]
    return precomp_alpha,precumprod[:,:,:,-1,:]

def Rasterize(shader ,target_camera, light, triplet:TripletModel,pipe, bg_color:torch.Tensor,scaling=1.0):

    raster_settings = RasterizationSettings(
        image_size=(int(target_camera.image_height*scaling), int(target_camera.image_width*scaling)), 
        faces_per_pixel=triplet.pixel_per_faces)
    rasterizer = MeshRasterizer(raster_settings=raster_settings
                                )
    # rasterizer = MeshRasterizerOpenGL(raster_settings=raster_settings)
    # verts = triplet.get_verts
    mesh = triplet.get_mesh
    faces = triplet.get_faces
    fragment,ndc_grad = rasterizer(mesh,cameras=target_camera.camera,eps=1e-15)
    pixel_colors,pixel_normal,vindex = shader(fragment,
                          mesh,
                          cameras=target_camera.camera,
                          lights=light,
                          materials = triplet.get_materials)

    pixel_alpha,bg_alpha = get_alpha_blend_pixel(triplet.get_alpha,faces,fragment)
    pixel_colors = (pixel_colors*pixel_alpha).sum(dim=-2)  
    alpha_mask = pixel_alpha.sum(-2)


    pixel = pixel_colors  + torch.ones_like(pixel_colors) * bg_color *  bg_alpha 
    pixel = pixel.permute(0,3,1,2) #BCHW
    
    pseudo_normal = get_pseudo_normal_from_depth(fragment,target_camera,pixel_alpha)
    with torch.no_grad():
       
        # pseudo_normal = get_pseudo_normal_from_depth(fragment,target_camera)
        # normal_img = get_normals_from_fragments(mesh, fragment)[..., 0,:]
        depth = fragment.zbuf[0, :, :, 0].detach().clone()*alpha_mask[0,...,0]
        # depth[depth==-1] = 100.
        depth = depth.cpu().numpy()

        depth_color = np_depth_to_colormap(depth)
                       
    return {"render": pixel[0],
            "visibility_filter" : [vindex[0],vindex[1]],
            "area": vindex[2],
            "depth_map":depth_color,
            "normal_map": (pixel_normal * pixel_alpha).sum(dim=-2).permute(0,3,1,2),
            "pseudo_normal":pseudo_normal*alpha_mask[0].permute(2,0,1),
            "alpha_channel":alpha_mask,
            'ndc_grad':ndc_grad}
