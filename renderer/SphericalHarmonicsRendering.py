from pytorch3d.renderer import (RasterizationSettings,
                                MeshRasterizer,
                                HardPhongShader,)
from scene.Triplet import TripletModel
import torch
from pytorch3d.ops import interpolate_face_attributes
from .render_utils import *
from .SH import *
max_degree = 3

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

def SHrender(target_camera, light, triplet:TripletModel,pipe, bg_color:torch.Tensor,FACES_PER_PIXEL,scaling=1.0,test=False,extent=10.):

    raster_settings = RasterizationSettings(
        image_size=(int(target_camera.image_height*scaling), int(target_camera.image_width*scaling)), 
        faces_per_pixel=triplet.pixel_per_faces)
    rasterizer = MeshRasterizer(raster_settings=raster_settings
                                )
    verts = triplet.get_verts
    mesh = triplet.get_mesh
    faces = triplet.get_faces
    fragment,verts_ndc = rasterizer(mesh,cameras=target_camera.camera,eps=1e-15)
    p2f = fragment.pix_to_face

    p2f = p2f.unsqueeze(dim = -1)
    p2f = p2f[p2f!=-1].reshape(-1) # the code of faces used
    # p2f_unique = p2f.unique() # unique code of faces used
    p2f_unique, area_counting = torch.unique(p2f,return_counts = True)
    pixel_face_mask = torch.zeros(faces.shape[0],dtype = torch.bool)
    pixel_face_mask[p2f_unique] = True 
    used_vertexes = faces[pixel_face_mask] # used vertexes,but wrap in faces 
    used_vertexes_unique = used_vertexes.reshape(-1).unique()      


    pixel_alpha,bg_alpha = get_alpha_blend_pixel(triplet.get_alpha,faces,fragment)



    SHC_Used,_ = triplet.get_SH_Coef
    SHC_Used = SHC_Used[:,used_vertexes.reshape(-1),...]
    positions =  verts[used_vertexes]
    positions = positions.reshape(-1,3).unsqueeze(dim=0).contiguous()
    color = SphereHarmonic.apply(positions.cuda(),triplet.sh_degree,(max_degree+1)**2,target_camera.camera.get_camera_center(),SHC_Used)
    color = color.reshape(positions.shape[0],-1,3,3) # batch_size, used_faces, the vert of the face, rgb
    colors = torch.zeros(faces.shape[0],3,3,device='cuda')

    colors[p2f_unique] = color
    face_attributes = colors
    F, FV, D = face_attributes.shape
    N, H, W, K, _ = fragment.bary_coords.shape
    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    pixel_vals =  interpolate_face_attributes(fragment.pix_to_face,
                                                fragment.bary_coords,
                                                face_attributes)
    # mask = fragment.pix_to_face < 0
    # pix_to_face = fragment.pix_to_face.clone()
    # pix_to_face[mask] = 0
    # idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D) # (...,3,D) <=>(...,three vertexes, dim of attribute)
    # pixel_face_vals = face_attributes.gather(0, idx).view(N, H, W, K, 3, D)
    # pixel_vals = (fragment.bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
    # # pixel_vals[mask] = backgroundcolor  # Replace masked values in output.
    # pixel_vals = pixel_vals.squeeze(dim = 3)
    materials_ambient = interpolate_face_attributes(
        fragment.pix_to_face, fragment.bary_coords, triplet.get_materials.ambient_color[:,faces][0]
    )
    ambient_color = light.ambient_color.view(1,1,1,1,-1) * materials_ambient
    pixel_vals += ambient_color
    pixel_colors = (pixel_vals*pixel_alpha).sum(dim=-2) 
    # pixel_colors = pixel_colors.permute(0,3,1,2) #BCHW
    # pixel_colors = pixel_colors + pixel_vals

    alpha_mask = pixel_alpha.sum(-2)

    # if target_alpha is not None:
    #     target = target_rgb[j] + (1-target_alpha[j]) * torch.ones_like(target_rgb[j])* backgroundcolor
    #     pixel = pixel_colors + torch.ones_like(pixel_colors)* backgroundcolor *  precumprod[:,:,:,-1,:]
    # else:
    #     target = target_rgb[j] 
    if test:
        pixel = pixel_colors  + torch.ones_like(pixel_colors) * torch.tensor(1.) *  bg_alpha 
    else:

        pixel = pixel_colors  + torch.ones_like(pixel_colors) * bg_color *  bg_alpha 

    # target = target.unsqueeze(0).permute(0,3,1,2)
    pixel = pixel.permute(0,3,1,2) #BCHW
    pseudo_normal = get_pseudo_normal_from_depth(fragment,target_camera,pixel_alpha)
    normal_img = get_normals_from_fragments(mesh, fragment)[..., 0,:]

    with torch.no_grad():
       
        # pseudo_normal = get_pseudo_normal_from_depth(fragment,target_camera)
        # normal_img = get_normals_from_fragments(mesh, fragment)[..., 0,:]
        depth = fragment.zbuf[0, :, :, 0].detach().clone()
        depth[depth==-1] = 100.
        depth = depth.cpu().numpy()

        depth_color, depth_normalized = np_depth_to_colormap(depth,max_depth= extent*2+1.)
                       
    return {"render": [pixel[0],torch.zeros_like(pixel[0])],
            "visibility_filter" : [p2f_unique,used_vertexes_unique],
            "area": area_counting,
            "depth_map":depth_color,
            "normal_map": normal_img,
            "pseudo_normal":pseudo_normal,
            "alpha_channel":alpha_mask,
            'verts_view':verts_ndc}
