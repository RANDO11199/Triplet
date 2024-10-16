# Copyright (C) 2024, Jiajie Yang https://github.com/RANDO11199
# All rights reserved.
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
import os
import random
import json
# from utils.system_utils import searchForMaxIteration
from .dataset_readers import sceneLoadTypeCallbacks
from .Triplet import TripletModel
from arguements import ModelParams,LightParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from pytorch3d.renderer import PointLights,DirectionalLights,MeshRasterizer,HardPhongShader,RasterizationSettings,EnvMapLights,SphericalHarmonics,SpectralModel,SphericalHarmonics_CUDA,SHLights,TexturesVertex
from pytorch3d.ops import interpolate_face_attributes,knn_points
from pytorch3d.structures import Meshes
import torch.nn as nn
import open3d as o3d
import numpy as np
import time
# from .dataset_readers import IntrinsicParams
import torch

class Scene:
    triplets: TripletModel

    def __init__(self, 
                args : ModelParams,
                LightArgs: LightParams,
                triplets : TripletModel,
                load_iteration=None, 
                camera_extent=None,
                shuffle=True,
                resolution_scales=[1.0]):
        
        self.model_path = args.model_path
        self.loaded_iter = None
        self.triplets = triplets

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        self.train_cameras = {}
        self.test_cameras = {}
        self.light = []
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.cameras_center = -scene_info.nerf_normalization["translate"]
        self.resolution_scales= resolution_scales
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        self.setupLighting(LightArgs,len(camlist),self.cameras_extent)

        if self.loaded_iter:
            self.triplets.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:

            self.triplets.create_from_pcd(scene_info.point_cloud,args,LightArgs,self.cameras_extent)
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.triplets.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # torch.save({
        #     'light_ambient_color':self._light_ambient_color,
        #     'light_diffuse_color':self._light_diffuse_color,
        #     'light_specular_color':self._light_specular_color,
        #     'intensity':self._light_intensity,
        #     ''
        # })
    def capture(self):
        return (self._light_ambient_color,self._light_ambient_intensity,
                self._light_diffuse_color,self._light_specular_color,self._light_intensity,
                self._light_pos, self.optimizer.state_dict())
    def restore(self,scene_params,training_args):
        (self._light_ambient_color,self._light_ambient_intensity,
        self._light_diffuse_color,self._light_specular_color,self._light_intensity,
        self._light_pos, opt_dict) = scene_params
        self.setupTraining(training_args)
        self.optimizer.load_state_dict(opt_dict)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def setupLighting(self,LightArgs,num_views=None,camera_extent = None,**kwargs):

        self._light_ambient_color = nn.Parameter(torch.tensor((LightArgs.light_ambient_color,),dtype=torch.float32),requires_grad=True)
        self._light_diffuse_color = nn.Parameter(torch.tensor((LightArgs.light_diffuse_color,),dtype=torch.float32)
                                                                                ,requires_grad=True)
        self._light_specular_color = nn.Parameter(torch.tensor((LightArgs.light_specular_color,),dtype=torch.float32)
                                                                                ,requires_grad=True)
        
        intensity = LightArgs.light_intensity if LightArgs.light_intensity is not None else (camera_extent**2,camera_extent**2,camera_extent**2)
        self._light_intensity = nn.Parameter(torch.tensor((intensity,),dtype=torch.float32),requires_grad=True)


        self._light_ambient_intensity = nn.Parameter(torch.tensor((LightArgs.ambient_intensity,),dtype=torch.float32),requires_grad=True) 

        self._light_pos = nn.Parameter(torch.tensor((self.cameras_center,),dtype=torch.float32),requires_grad=True) # Fixed location 
        self._light_direction = nn.Parameter(torch.tensor(((0.1,1.,0.1),),dtype=torch.float32),requires_grad=True)# Fixed location 

        self.sh = SphericalHarmonics(LightArgs.envmap_resolution, device='cuda:0')
        self.diffuse_sh_coefs = torch.randn((1, LightArgs.diffuse_band, 3), device='cuda:0')  # white light for a while 
        self.diffuse_sh_coefs.requires_grad = True
        self.specular_sh_coefs = torch.randn((1, LightArgs.specular_band, 3), device='cuda:0') # white light for a while 
        self.specular_sh_coefs.requires_grad = True

        # self._light_pos = nn.Parameter(self._light_pos,requires_grad = True)
        self.setupTraining(LightArgs)
        self.LStype = LightArgs.LStype

    def setupTraining(self,LightArgs):
        l = [
            {'params':self._light_pos,"name":'light_position','lr':LightArgs.light_position_lr},
            {'params':self._light_direction,"name":'light_position','lr':LightArgs.light_direction_lr},
            {'params':self._light_ambient_color,"name":'light_ambient','lr':LightArgs.light_ambient_lr},
            {'params':self._light_diffuse_color,"name":'light_difffuse','lr':LightArgs.light_diffuse_lr},
            {'params':self._light_specular_color,"name":'light_specular','lr':LightArgs.light_specular_lr},
            {'params':self._light_intensity,"name":'light_intensity','lr':LightArgs.light_intensity_lr},
            {'params':self._light_ambient_intensity,"name":'light_ambient_intensity','lr':LightArgs.light_ambient_intensity_lr},                          
            {'params':self.diffuse_sh_coefs,"name":'diffuse_sh_coefs','lr':LightArgs.diffuse_sh_coefs_lr}, 
            {'params':self.specular_sh_coefs,"name":'specular_sh_coefs','lr':LightArgs.specular_sh_coefs_lr}, 

            ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def getLightSource(self,ind =None):
        if self.LStype == 'Point':
            return PointLights(device='cuda',
                                ambient_color=torch.clamp(self._light_ambient_color,min=0.,max=1.),
                                diffuse_color=torch.clamp(self._light_diffuse_color,min=0.,max=1.),
                                specular_color=torch.clamp(self._light_specular_color,min=0.,max=1.),
                                intensity=torch.nn.functional.relu(self._light_intensity),
                                ambient_intensity=torch.nn.functional.relu(self._light_ambient_intensity),
                                location=self._light_pos),None,None
        elif self.LStype =='Direction':
            return DirectionalLights(device='cuda',
                                ambient_color=torch.clamp(self._light_ambient_color,min=0.,max=1.),
                                diffuse_color=torch.clamp(self._light_diffuse_color,min=0.,max=1.),
                                specular_color=torch.clamp(self._light_specular_color,min=0.,max=1.),
                                intensity=torch.nn.functional.relu(self._light_intensity),
                                ambient_intensity=torch.nn.functional.relu(self._light_ambient_intensity),
                                direction=self._light_direction
                                ), None,None
        elif self.LStype =='EnvMap':
            diffuse_envmap = self.sh.toEnvMap(self.diffuse_sh_coefs).clip(0, 1)
            specular_envmap = self.sh.toEnvMap(self.specular_sh_coefs).clip(0, 1)
            # diffuse_envmap = self.sd().clip(0, 1).permute(0,2,3,1).float().contiguous()
            # specular_envmap = self.sp().clip(0,1).permute(0,2,3,1).float().contiguous()
            if not self.specular_sh_coefs.requires_grad or not self.diffuse_sh_coefs.requires_grad:
                diffuse_envmap = torch.ones_like(diffuse_envmap )*0.7
                specular_envmap = torch.ones_like(specular_envmap)*0.7
            return EnvMapLights(device='cuda',
                                ambient_color = torch.clamp(self._light_ambient_color,min=0.,max=1.),
                                diffuse_color=diffuse_envmap,
                                specular_color=specular_envmap,
                                intensity=torch.nn.functional.relu(self._light_intensity),
                                ambient_intensity=torch.nn.functional.relu(self._light_ambient_intensity)), diffuse_envmap, specular_envmap
        elif self.LStype == 'SH':
            return self.triplets.get_vert_based_SH_Light(ambient_color=self._light_ambient_color),None,None
    def SceneSurfaceReconstruction(self,):
        pass

    @torch.no_grad()
    def get_cam_open3d(self, target_cameras,H=800,W=800):
        o3d_target_camera = []
        for i, viewpoint_cam in enumerate(target_cameras):
            # Handle the handness of camera # Need to be Rright-Down-Forward
            w2c = target_cameras[i].get_world_to_view_transform().get_matrix()[0].T
            c2w = torch.linalg.inv(w2c)
            R, T = c2w[:3, :3], c2w[:3, 3:]
            R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1) # from RDF to Left-Up-Forward for Rotation
            new_c2w = torch.cat([R, T], 1)
            extrinsic = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]]).to('cuda')), 0))

            # extrinsic = (target_cameras[i].get_world_to_view_transform().get_matrix()[0]).cpu().numpy()
            intrins = ((target_cameras[i].get_projection_transform().get_matrix()[:3,:3]).T).cpu().numpy()

            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=W,
                height=H,
                cx = intrins[0,2].item(),
                cy = intrins[1,2].item(), 
                fx = intrins[0,0].item(), 
                fy = intrins[1,1].item()
            )
            camera = o3d.camera.PinholeCameraParameters()
            camera.extrinsic = extrinsic.cpu().numpy()
            camera.intrinsic = intrinsic
            o3d_target_camera.append(camera)

        return o3d_target_camera

    def TSDF(self,shader,voxel_size=0.04,sdf_trunc=0.02):
        # From GS2D, temporarily. Enough for object but not for scene
        camera_o3d = self.get_cam_open3d(self.train_cameras, H=self.train_cameras[0].image_height,W=self.train_cameras[0].image_width)
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        faces = self.triplets.get_faces
        face_attribute_alpha = self.triplets.get_alpha[faces].unsqueeze(-1)
        rasterization_setting = RasterizationSettings((self.train_cameras[0].image_height,self.train_cameras[0].image_width),
                                                      faces_per_pixel=self.triplets.pixel_per_faces)
        rasterizer = MeshRasterizer(rasterization_setting)
        mesh = self.triplets.get_mesh
        for j in range(len(self.train_cameras)):
            materials = self.triplets.get_materials
            fragments = rasterizer(mesh,cameras=self.train_cameras[j].camera)

            face_attribute = interpolate_face_attributes(fragments.pix_to_face,
                                                            fragments.bary_coords,
                                                            face_attribute_alpha)
            pixel_alphas = face_attribute
            precumprod = torch.cumprod(1- torch.cat([torch.zeros([pixel_alphas.shape[0],
                                                                    pixel_alphas.shape[1],
                                                                    pixel_alphas.shape[2],1,1],
                                                                    device=pixel_alphas.device),
                                                                    pixel_alphas],dim=3) + 1e-10,dim=3)
            precomp_alpha =  pixel_alphas * precumprod[:,:,:,:-1,:]

            depth = (fragments.zbuf * precomp_alpha[...,0]).sum(-1).permute(1,2,0)

            # depth = fragments.zbuf[...,0].permute(1,2,0)

            # if true_alpha is not None:
            #     depth[true_alpha[j] < 0.5] =  0.

            images_predicted = shader(fragments,
                                    mesh,
                                    cameras=self.train_cameras[j].camera,
                                    materials = materials,
                                    lights=self.getLightSource())
            
            pixel_colors =images_predicted
            pixel_colors = (pixel_colors*precomp_alpha).sum(dim=-2)[0]
            
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(pixel_colors.cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.cpu().numpy(), order="C")),
                depth_trunc = self.cameras_extent*2 , convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )
            volume.integrate(rgbd, intrinsic=camera_o3d[j].intrinsic, extrinsic=camera_o3d[j].extrinsic)
            torch.cuda.empty_cache()

        mesh = volume.extract_triangle_mesh()
        return mesh
    
    def post_process_mesh(self,mesh, cluster_to_keep=100,voxel_size = 32, iteration= -1):
        """
        Post-process a mesh to filter out floaters and disconnected parts
        """
        import copy
        print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
        mesh_0 = copy.deepcopy(mesh)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        n_cluster = np.sort(cluster_n_triangles.copy())
        if n_cluster.shape[0] > cluster_to_keep:
            n_cluster = n_cluster[-cluster_to_keep]
        else:
            n_cluster = n_cluster[-n_cluster.shape[0]]
        n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        voxel_size = max(mesh_0.get_max_bound() - mesh_0.get_min_bound()) / voxel_size
        # if i !=-1 and i < 10000:
        #     mesh_0 = mesh_0.simplify_vertex_clustering(voxel_size,
        #                                    contraction=o3d.geometry.SimplificationContraction.Average) # faster
        # else:

        mesh_0 = mesh_0.simplify_quadric_decimation(target_number_of_triangles = len(mesh_0.vertices)//20 ) #

        mesh_0.remove_unreferenced_vertices()
        mesh_0.remove_degenerate_triangles()
        # mesh_0.fill_holes()
        print("num vertices raw {}".format(len(mesh.vertices)))
        print("num vertices post {}".format(len(mesh_0.vertices)))
        return mesh_0
    
    def surface_extraction_postfix(self,mesh_o3d_post,opt):

        mesh_o3d_post.compute_vertex_normals()

        verts_sdf = torch.from_numpy(np.asarray(mesh_o3d_post.vertices)).float().to('cuda')

        faces_sdf = torch.from_numpy( np.asarray(mesh_o3d_post.triangles)).float().to('cuda')

        # texts = torch.from_numpy( np.asarray(mesh_o3d_post.vertex_colors)).float().to('cuda').unsqueeze(0)

        # verts_normal_sdf = torch.from_numpy(np.asarray(mesh_o3d_post.vertex_normals)).float().to('cuda')

        # verts_sdf = verts_sdf[0]/scale + center
        # faces_sdf = faces_sdf[0]
        verts = self.triplets.get_verts
        texts = self.triplets.get_textures('raw')
        dists, idx, nn_ = knn_points(verts_sdf.unsqueeze(0).float().cuda(),
                                        verts.unsqueeze(0).float().cuda(),K=1)
        screen_area = torch.zeros(faces_sdf.shape[0]).to('cuda')

        texts = texts[:,idx[0,:,0],:].float().to('cuda')
        ambient_materials = self.triplets._ambient_materials[:,idx[0,:,0],:].float().to('cuda')
        diffuse_materials = self.triplets._diffuse_materials[:,idx[0,:,0],:].float().to('cuda')
        specular_materials = self.triplets._specular_materials[:,idx[0,:,0],:].float().to('cuda')
        
        shininess = self.triplets._shininess[:,idx[0,:,0],:].float().to('cuda')
        src_mesh = Meshes(verts=verts_sdf.unsqueeze(0),faces=faces_sdf.unsqueeze(0))
                            # ,verts_normals=verts_normal_sdf.unsqueeze(0))
        src_mesh.textures = TexturesVertex(verts_features=texts) 

        deform_verts = torch.full(verts_sdf.shape, 0.0, device='cuda', requires_grad=True)
        alpha = self.triplets._alpha[:,idx[0,:,0],:].float().to('cuda')

        optimizer = torch.optim.Adam([
                    {'params':deform_verts,'lr':opt.deform_lr,'name':"deform"},
                    {'params':texts,"name":'textures','lr':opt.texts_lr},
                    {'params':alpha,"name":'alpha','lr':opt.alpha_lr},
                    {'params':ambient_materials,"name":'ambient_materials','lr':opt.ambient_materials_lr},
                    {'params':diffuse_materials,"name":'diffuse_materials','lr':opt.diffuse_materials_lr},
                    {'params':specular_materials,"name":'specular_materials','lr':opt.specular_materials_lr},
                    {'params':shininess,"name":'shininess','lr':opt.shininess_lr},
                    ],lr=0.0,betas=(0.5,0.9),eps=1e-12)
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        # new_src_mesh = taubin_smoothing(new_src_mesh,num_iter=30)
        src_mesh = new_src_mesh.offset_verts(-deform_verts)
        deform_verts_grad_accum =  torch.zeros((deform_verts.shape[0], 1), device="cuda")
        denom_grad = torch.zeros((deform_verts.shape[0], 1), device="cuda")
        alpha_accum = torch.zeros((deform_verts.shape[0], 1), device="cuda")
        denom_alpha = torch.zeros((deform_verts.shape[0], 1), device="cuda")

    def extract_mesh_from_scene(self,shader,opt):
        if opt.method=="TSDF":
            mesh = self.TSDF(shader)
            post_mesh = self.post_process_mesh(mesh)
            self.surface_extraction_postfix(post_mesh,opt)
    def check_progress(self,iteration):
        if iteration > 1000:
            self._light_ambient_color.requires_grad = True
        else:
            self._light_ambient_color.requires_grad = False


