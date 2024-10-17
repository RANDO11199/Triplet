from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
import os
from utils.graphics_utils import BasicPointCloud
import torch
import numpy as np
from pytorch3d.ops import knn_points,SubdivideMeshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Materials,TexturesVertex,SHLights
from pytorch3d.io import save_ply
import torch.nn as nn 
import os
from utils.general_utils import inverse_sigmoid, get_expon_lr_func,get_step_lr_func
from utils.system_utils import mkdir_p
from plyfile import PlyElement,PlyData
from renderer.SH import RGB2SH
class TripletModel:
    # Triangular Patchlet
    def setup_functions(self):
        self.act = torch.sigmoid
        self.deact = inverse_sigmoid

    def __init__(self):

        # Position and transparency
        self._alpha = torch.empty(0)
        self._deform_verts = torch.empty(0)
        self.deform_verts_grad_accum = torch.empty(0)
        self.denom = torch.empty(0)
        # Materials
        self._ambient_materials = torch.empty(0)
        self._diffuse_materials = torch.empty(0)
        self._specular_materials = torch.empty(0)
        self._emission_materials = torch.empty(0)
        self._shininess = torch.empty(0)
        # Light Compensation
        self._texts = torch.empty(0)
        # Optimization
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.optimizer = None
        self.alpha_reset_val = 0.01
        self.setup_functions()
        self.view_dependent_compensation = False
        self.min_alpha = 0.05
        self.resolution_scale = 1.
        self.pixel_per_faces=10
        self.sh_indicator = True
        self.count_sh = 0
        self.sh_degree = 0 # 0
    @property
    def get_alpha(self):
        return self.act(self._alpha)
    @property
    def get_materials(self):
        return Materials(ambient_color = self.get_ambient_materials,
                            diffuse_color = self.get_diffuse_materials,
                            specular_color = self.get_specular_materials,
                            shininess = self.get_shininess, # margin to 0., -0. cause overflow i.e. 1/-0.
                            device = self._deform_verts.device)

    def get_textures(self,texts_format=None):
        
        if texts_format == 'vertex':
            return TexturesVertex(verts_features=self._texts.clamp(min=0.,max=1.))
        elif texts_format == 'UV':
            raise NotImplementedError('TexturesUV not supported yet')
        elif texts_format=='raw':
            return self._texts.clamp(min=0.,max=1.)
    @property
    def get_vert_normals(self):
        return self._src_mesh.verts_normals_packed()
    @property
    def get_verts(self):
        return self._src_mesh.offset_verts(self._deform_verts).verts_packed()
    @property
    def get_faces(self):
        return self._src_mesh.faces_packed()
    @property
    def get_mesh(self):
        self._src_mesh.textures = self.get_textures('vertex')
        return self._src_mesh.offset_verts(self._deform_verts)
    @property
    def get_deform(self):
        return self._deform_verts
    @property
    def get_ambient_materials(self):
        return self.act(self._ambient_materials)
    @property
    def get_diffuse_materials(self):
        return self.act(self._diffuse_materials)
    @property
    def get_specular_materials(self):
        return self.act(self._specular_materials)
    @property
    def get_shininess(self):
        if self.materials_type =='CT':
            return self.act(self._shininess)
        elif self.materials_type =='BP':
            return self._shininess
    @property
    def get_emission_materials(self):
        return self._emission_materials
    def capture(self):
        return (
        # Position and transparency
        self._alpha,
        self.get_verts,
        self.get_faces,
        self.deform_verts_grad_accum,
        self.denom,
        self.max_screen_area,
        # Materials
        self._ambient_materials,
        self._diffuse_materials,
        self._specular_materials,
        self._shininess,
        # Light Compensation
        self._texts,
        # Optimization
        self.spatial_lr_scale,
        self.optimizer.state_dict()
        )
    def restore(self, model_args, training_args):
        (
        self._alpha,
        self._deform_verts,
        deform_verts_grad_accum,
        denom,
        # Materials
        self._ambient_materials,
        self._diffuse_materials,
        self._specular_materials,
        self._emission_materials,
        self._shininess,
        # Light Compensation
        self._texts,
        # Optimization
        self.spatial_lr_scale,
        opt_dict) = model_args
        self.training_setup(training_args)
        self.deform_verts_grad_accum = deform_verts_grad_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    
    
    def create_from_pcd(self, pcd : BasicPointCloud,opt, LightArgs, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).reshape(1,-1,3).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).reshape(1,-1,3).float().cuda()
        # create company for lonely point
        # save lonely point
        
        # new_point_1 = (( ( 0.1* spatial_lr_scale * (torch.randn(1,fused_point_cloud.shape[1],local_times[0],3,device='cuda') ) )) + fused_point_cloud.reshape(1,-1,1,3).repeat(1,1,local_times[0],1)).reshape(1,-1,3)
        # new_texts_1 = fused_color.reshape(1,-1,1,3).repeat(1,1,local_times[0],1).reshape(1,-1,3)

        # new_point_2 = (( ( 0.3*  spatial_lr_scale * (torch.randn(1,fused_point_cloud.shape[1],local_times[1],3,device='cuda') ) )) + fused_point_cloud.reshape(1,-1,1,3).repeat(1,1,local_times[1],1)).reshape(1,-1,3)
        # new_texts_2 = fused_color.reshape(1,-1,1,3).repeat(1,1,local_times[1],1).reshape(1,-1,3)

        # new_point_3 = (( ( 0.5*  spatial_lr_scale * (torch.randn(1,fused_point_cloud.shape[1],local_times[2],3,device='cuda') ) )) + fused_point_cloud.reshape(1,-1,1,3).repeat(1,1,local_times[2],1)).reshape(1,-1,3)
        # new_texts_3 = fused_color.reshape(1,-1,1,3).repeat(1,1,local_times[2],1).reshape(1,-1,3)

        # On for Scene
        if LightArgs.compensate_random_Point:
            local_times = 8 # 11 is good if vram enoughn
            new_point_4 = (( ( spatial_lr_scale * (torch.randn(1,fused_point_cloud.shape[1],local_times,3,device='cuda') ) )) + fused_point_cloud.reshape(1,-1,1,3).repeat(1,1,local_times,1)).reshape(1,-1,3)
            new_texts_4 = fused_color.reshape(1,-1,1,3).repeat(1,1,local_times,1).reshape(1,-1,3)
            fused_point_cloud = torch.cat([fused_point_cloud,new_point_4],dim= 1)
            fused_color = torch.cat([fused_color,new_texts_4],dim = 1 )

        #position and transparency
        deform_verts = torch.full(fused_point_cloud.shape[1:], 0.0, device='cuda', requires_grad=True)
        alpha = inverse_sigmoid(0.1* torch.ones((fused_point_cloud.shape[1]), dtype=torch.float, device="cuda"))

        # Materials
        # Compensation
        dists, idx, nn_ = knn_points(fused_point_cloud,fused_point_cloud,K=3)
        
        self.max_screen_area = torch.zeros(idx.shape[1]).to('cuda')
        self._src_mesh = Meshes(fused_point_cloud,idx)
        self.max_sh_degree = LightArgs.max_sh_degree
        print("Number of verts at initialisation : ", fused_point_cloud.shape[1], "Number of faces at initialisation : ", idx.shape[1])
        if opt.renderer == "Rasterization":
            if opt.materials_type=="BP":
                ambient_materials = torch.tensor([[[0.2,0.2,0.2],],]).cuda() * fused_color
                diffuse_materials = torch.tensor([[[0.5,0.5,0.5],],]).cuda() * fused_color
                specular_materials = torch.tensor([[[0.7,0.7,0.7],],]).cuda() * fused_color
                shininess = nn.Parameter(64.* torch.ones(1,fused_point_cloud.shape[1],1,device='cuda').requires_grad_(True))
                texts = (192./255.)*(torch.ones(1,fused_point_cloud.shape[1],3,device='cuda'))
                self._alpha = nn.Parameter(alpha,requires_grad=True)
                self._ambient_materials = nn.Parameter(ambient_materials.contiguous(),requires_grad=True)
                self._diffuse_materials = nn.Parameter(diffuse_materials.contiguous(),requires_grad=True)
                self._specular_materials = nn.Parameter(specular_materials.contiguous(),requires_grad=True)
                self._shininess = nn.Parameter(shininess.contiguous(),requires_grad=True)
                self._texts = nn.Parameter(fused_color.contiguous(),requires_grad=True)
                self._deform_verts = nn.Parameter(deform_verts.contiguous(),requires_grad = True)
                SphereHarmonics_Coeffs = torch.zeros(1,fused_color.shape[1],6,(self.max_sh_degree+1)**2)
                self._feature_dc_s = nn.Parameter(torch.cat([RGB2SH(torch.ones_like(fused_color)).contiguous().requires_grad_(True).unsqueeze(-1),(torch.tensor([0.1,0.1,0.1])*torch.randn_like(fused_color)).unsqueeze(-1) ],dim=-2)).cuda()
                self._feature_rest_s = nn.Parameter(SphereHarmonics_Coeffs[...,1:].contiguous().requires_grad_(True)).cuda()
                self.materials_type = 'BP'
                del SphereHarmonics_Coeffs
            if opt.materials_type=="CT":                                    
                ambient_materials = self.deact(torch.clamp(torch.tensor([[[1.0],],]).cuda()*torch.ones(1,fused_point_cloud.shape[1],1,device='cuda') + 0.05 * (torch.randn(1,fused_point_cloud.shape[1],1,device='cuda')),0.,1.))# AO
                diffuse_materials = self.deact(fused_color) #albedo
                # Q: Why 0.1 0.9?
                # A: decrease the sensitivity to lights at the very beginning of train. Specular light is more sensitive
                # Channel 1 is traditionally used / channel 3 better result
                # if dim3 roughness lr = 0.001
                # metallic = self.deact(torch.clamp(torch.tensor([[[0.1,0.1,0.1],],]).cuda()*(torch.ones(1,fused_point_cloud.shape[1],3,device='cuda')) + 0.05 * (torch.randn(1,fused_point_cloud.shape[1],3,device='cuda')),0.,1.)) # METALLIC
                # roughness = self.deact(torch.clamp(torch.tensor([[[0.9,0.9,0.9],],]).cuda()*(torch.ones(1,fused_point_cloud.shape[1],3,device='cuda')) + 0.05 * (torch.randn(1,fused_point_cloud.shape[1],3,device='cuda')),min=0.,max=1.))
                # if dim1 roughness lr = 0.01
                metallic = self.deact(torch.clamp(torch.tensor([[[0.1],],]).cuda()*(torch.ones(1,fused_point_cloud.shape[1],1,device='cuda')) + 0.05 * (torch.randn(1,fused_point_cloud.shape[1],1,device='cuda')),0.,1.)) # METALLIC
                roughness = self.deact(torch.clamp(torch.tensor([[[0.9],],]).cuda()*(torch.ones(1,fused_point_cloud.shape[1],1,device='cuda')) + 0.05 * (torch.randn(1,fused_point_cloud.shape[1],1,device='cuda')),min=0.,max=1.))
                self._alpha = nn.Parameter(alpha,requires_grad=True)
                self._ambient_materials = nn.Parameter(ambient_materials.contiguous(),requires_grad=True)
                self._diffuse_materials = nn.Parameter(diffuse_materials.contiguous(),requires_grad=True)
                self._specular_materials = nn.Parameter(metallic.contiguous(),requires_grad=True)
                self._shininess = nn.Parameter(roughness.contiguous(),requires_grad=True)
                self._texts = nn.Parameter((torch.ones(1,fused_point_cloud.shape[1],1,device='cuda')) ,requires_grad=True)
                self._deform_verts = nn.Parameter(deform_verts.contiguous(),requires_grad = True)
                SphereHarmonics_Coeffs = torch.zeros(1,fused_color.shape[1],6,(self.max_sh_degree+1)**2)
                self._feature_dc_s = nn.Parameter(torch.cat([RGB2SH(torch.ones_like(fused_color)).contiguous().requires_grad_(True).unsqueeze(-1),(torch.tensor([0.1,0.1,0.1])*torch.randn_like(fused_color)).unsqueeze(-1) ],dim=-2)).cuda()
                self._feature_rest_s = nn.Parameter(SphereHarmonics_Coeffs[...,1:].contiguous().requires_grad_(True)).cuda()
                self.materials_type = 'CT'
                del SphereHarmonics_Coeffs
        
    @property
    def get_SH_Coef(self):
        return torch.cat([self._feature_dc_s,self._feature_rest_s],dim =-1 )

    def get_vert_based_SH_Light(self,ambient_color):
        specular_coef = self.get_SH_Coef
        return SHLights(specular_color=specular_coef,sh_degree = self.sh_degree,device='cuda',ambient_color=ambient_color,max_sh_degree=self.max_sh_degree)
    def training_setup(self,training_args):
        self.percent_dense = training_args.percent_dense
        self.deform_verts_grad_accum = torch.zeros((self._deform_verts.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._deform_verts.shape[0], 1), device="cuda")

        self.optimizer = torch.optim.Adam([
            {'params':self._deform_verts,'lr':training_args.deform_lr*self.spatial_lr_scale,'name':"deform"},
            {'params':self._texts,"name":'textures','lr':training_args.texts_lr},
            {'params':self._alpha,"name":'alpha','lr':training_args.alpha_lr},
            {'params':self._ambient_materials,"name":'ambient_materials','lr':training_args.ambient_materials_lr},
            {'params':self._diffuse_materials,"name":'diffuse_materials','lr':training_args.diffuse_materials_lr},
            {'params':self._specular_materials,"name":'specular_materials','lr':training_args.specular_materials_lr},
            {'params':self._shininess,"name":'shininess','lr':training_args.shininess_lr},
            {'params':self._feature_dc_s,"name":'feature_dc_s','lr':training_args.SH_lr},
            {'params':self._feature_rest_s,"name":'feature_rest_s','lr':training_args.SH_lr/10},
        ], lr=0.0,eps=1e-15,betas=(0.5,0.999))
        self.deform_verts_scheduler_args = get_expon_lr_func(lr_init=training_args.deform_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deform_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deform_lr_delay_mult,
                                                    max_steps=training_args.deform_lr_max_steps)
        self.materials_scheduler_args = get_step_lr_func(training_args.ambient_materials_lr,gamma=5.)               
        self.SR_scheduler_args = get_step_lr_func(training_args.shininess_lr,gamma=5.)               
        self._feature_dc_scheduler_args = get_step_lr_func(training_args.SH_lr,gamma=5.)               
        self._feature_rest_scheduler_args = get_step_lr_func(training_args.SH_lr/10,gamma=5.)               
                                      
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        # self.scheduler.step()  
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_verts_scheduler_args(iteration)
                param_group['lr'] = lr
            elif ( param_group['name'] == 'ambient_materials'
                or param_group['name'] == 'diffuse_materials'
                or param_group['name'] == 'specular_materials'
                or param_group['name'] == 'textures'):
                lr = self.materials_scheduler_args(iteration)
                param_group['lr'] = lr
            elif ( param_group['name'] == 'shininess'
                    ):
                lr = self.SR_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group['name'] == 'feature_dc_s':
                lr = self._feature_dc_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group['name'] == 'feature_dc_s':
                lr = self._feature_rest_scheduler_args(iteration)
                param_group['lr'] = lr
    # def construct_list_of_attributes(self):
    #     l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    #     # All channels except the 3 DC
    #     # for i in range(self.get_faces.shape[1]):
    #     #     l.append('faces_{}'.format(i))
    #     for i in range(self._texts.shape[2]):
    #         l.append('textures_{}'.format(i))
    #     for i in range(self._ambient_materials.shape[2]):
    #         l.append('ambient_materials_{}'.format(i))
    #     for i in range(self._diffuse_materials.shape[2]):
    #         l.append('diffuse_materials_{}'.format(i))
    #     for i in range(self._specular_materials.shape[2]):
    #         l.append('specular_materials_{}'.format(i))
    #     for i in range(self._emission_materials.shape[2]):
    #         l.append('emission_materials_{}'.format(i))
    #     l.append('alpha')
    #     l.append('shininess')
    #     return l
    # def set_mesh(self,verts,faces,mask,texts=None):
    #     if texts is not None:
    #         self._src_mesh = Meshes(verts.unsqueeze(0),faces[~mask].unsqueeze(0))
    #         self._src_mesh.textures = TexturesVertex(verts_features=texts) 

    def save_ply_geometry(self,iteration):
     
        inter_verts, inter_faces = self.get_mesh.get_mesh_verts_faces(0)
        
        inter_obj = f'./tests/model_geometry_step_{iteration}.ply'
        save_ply(
            f=inter_obj,
            verts=inter_verts,
            faces=inter_faces,
            verts_normals = self.get_mesh.verts_normals_packed(),
        )

    #TODO: Later
    def save_ply(self, path):
        # mkdir_p(os.path.dirname(path))
        # xyz = self.get_verts
        # faces = self.get_faces.detach().cpu().numpy()
        # normals = self.get_vert_normals
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # element_faces = np.empty(faces.shape[0], dtype=[('vertex_indices_0', 'i4'),
        #                                                 ('vertex_indices_1', 'i4'),
        #                                                 ('vertex_indices_2', 'i4')
        #                                                 ])

        # attributes = np.concatenate((xyz.detach().cpu().numpy(), 
        #                              normals.detach().cpu().numpy(), 
        #                              self.get_textures('raw')[0].detach().cpu().numpy(),
        #                              self.get_ambient_materials[0].detach().cpu().numpy(),
        #                              self.get_diffuse_materials[0].detach().cpu().numpy(),
        #                              self.get_specular_materials[0].detach().cpu().numpy(),
        #                              self._emission_materials[0].detach().cpu().numpy(),
        #                              self.get_alpha.unsqueeze(-1).detach().cpu().numpy(),
        #                              self.get_shininess[0].detach().cpu().numpy()), axis=1)
        # face = np.array(self.get_faces,
        #                     dtype=[('vertex_indices', 'i4', (3,))])

        # elements[:] = list(map(tuple, attributes))

        # element_faces[:] = list(map(tuple, faces))
        
        # el = PlyElement.describe(elements, 'vertex')
        # el = PlyElement.describe(element_faces, 'face',
        #                   val_types={'vertex_indices': 'u2'},
        #                   len_types={'vertex_indices': 'u4'})
        # PlyData([el]).write(path)
        pass

    def load_ply(self, path):
        # plydata = PlyData.read(path)

        # xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
        #                 np.asarray(plydata.elements[0]["y"]),
        #                 np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        # n_xyz = np.stack((np.asarray(plydata.elements[0]["nx"]),
        #                 np.asarray(plydata.elements[0]["ny"]),
        #                 np.asarray(plydata.elements[0]["nz"])),  axis=1)
        
        # faces = np.stack((np.asarray(plydata.elements[0]["faces_0"]),
        #                 np.asarray(plydata.elements[0]["faces_1"]),
        #                 np.asarray(plydata.elements[0]["faces_2"])),  axis=1)

        # alpha = np.asarray(plydata.elements[0]["alpha"])
        # shininess = np.asarray(plydata.elements[0]["shininess"])[..., np.newaxis]

        # texts = np.zeros((xyz.shape[0], 3))
        # texts[:, 0] = np.asarray(plydata.elements[0]["textures_0"])
        # texts[:, 1] = np.asarray(plydata.elements[0]["textures_1"])
        # texts[:, 2] = np.asarray(plydata.elements[0]["textures_2"])

        # ambient_materials = np.zeros((xyz.shape[0], 3))
        # ambient_materials[:, 0] = np.asarray(plydata.elements[0]["ambient_materials_0"])
        # ambient_materials[:, 1] = np.asarray(plydata.elements[0]["ambient_materials_1"])
        # ambient_materials[:, 2] = np.asarray(plydata.elements[0]["ambient_materials_2"])

        # diffuse_materials = np.zeros((xyz.shape[0], 3))
        # diffuse_materials[:, 0] = np.asarray(plydata.elements[0]["diffuse_materials_0"])
        # diffuse_materials[:, 1] = np.asarray(plydata.elements[0]["diffuse_materials_1"])
        # diffuse_materials[:, 2] = np.asarray(plydata.elements[0]["diffuse_materials_2"])

        # specular_materials = np.zeros((xyz.shape[0], 3))
        # specular_materials[:, 0] = np.asarray(plydata.elements[0]["specular_materials_0"])
        # specular_materials[:, 1] = np.asarray(plydata.elements[0]["specular_materials_1"])
        # specular_materials[:, 2] = np.asarray(plydata.elements[0]["specular_materials_2"])

        # emission_materials = np.zeros((xyz.shape[0], 3))
        # emission_materials[:, 0] = np.asarray(plydata.elements[0]["emission_materials_0"])
        # emission_materials[:, 1] = np.asarray(plydata.elements[0]["emission_materials_1"])
        # emission_materials[:, 2] = np.asarray(plydata.elements[0]["emission_materials_2"])

        # self._src_mesh = Meshes(torch.from_numpy(xyz).unsqueeze(0),torch.from_numpy(faces).unsqueeze(0))
        # self._deform_verts = nn.Parameter(torch.zeros_like(torch.from_numpy(xyz)),requires_grad=True)
        # self._ambient_materials = nn.Parameter(torch.from_numpy(ambient_materials).unsqueeze(0), requires_grad = True)
        # self._diffuse_materials = nn.Parameter(torch.from_numpy(diffuse_materials).unsqueeze(0), requires_grad = True)
        # self._specular_materials = nn.Parameter(torch.from_numpy(specular_materials).unsqueeze(0), requires_grad= True)
        # self._emission_materials = nn.Parameter(torch.from_numpy(emission_materials).unsqueeze(0),requires_grad= True)
        # self._shininess = nn.Parameter(torch.from_numpy(shininess).unsqueeze(0), requires_grad = True)
        # self._texts = nn.Parameter(torch.from_numpy(texts).unsqueeze(0),requires_grad=True)
        # self.alpha = nn.Parameter(torch.from_numpy(alpha),requires_grad=True)
        pass

    def kill_large_mesh(self,max_screen_size):

        large_mesh = (self.max_screen_area > max_screen_size )
        self._src_mesh =  Meshes(self._src_mesh.verts_packed().unsqueeze(0),self._src_mesh.faces_packed()[~large_mesh].unsqueeze(0))
        self.max_screen_area = self.max_screen_area[~large_mesh]

    def densify_and_prune(self, max_grad, min_alpha, scene_extent):
        grads = self.deform_verts_grad_accum/self.denom
        grads[grads.isnan()] = 0.0
        # max_grad = self.max_grad
        self.prune_large_mesh_world(scene_extent)

        verts_packed = self.get_verts  # (sum(V_n), 3)
        faces_packed = self.get_faces  # (sum(V_n), 3)
        verts_edges = verts_packed[faces_packed]
        v0, v1, v2 = verts_edges.unbind(1)
        l0, l1, l2 = (v0 - v1).norm(dim=1, p=2), (v0 - v2).norm(dim=1, p=2), (v1 - v2).norm(dim=1, p=2)
        length_max, _ = torch.stack([l0,l1,l2],dim=-1).max(dim=-1)

        deform_verts_grad_backup = self.get_deform.grad
        verts_grad = (self.deform_verts_grad_accum/self.denom).squeeze(-1)[self.get_faces]

        # Extract points that satisfy the gradient condition
        verts_max_grad, vert_max_indice = verts_grad.max(dim = -1)

        selected_pts_mask_clone =  torch.where(verts_max_grad >= max_grad, True, False)
        selected_pts_mask_split = torch.where(verts_max_grad >= max_grad, True, False) 

        selected_pts_mask_clone = torch.logical_and(selected_pts_mask_clone,
                                                length_max <= self.percent_dense * scene_extent) 

        selected_pts_mask_split = torch.logical_and(selected_pts_mask_split,
                                                length_max > self.percent_dense * scene_extent)
        self.densify_and_split_flexible(selected_pts_mask_split, scene_extent)

        padded_selected_pts_mask_clone = torch.zeros(self.get_faces.shape[0],dtype = torch.bool)
        padded_selected_pts_mask_clone[:selected_pts_mask_clone.shape[0]] = selected_pts_mask_clone
        self.densify_and_clone(padded_selected_pts_mask_clone, scene_extent)
        # TODO: Check: is it worked? The graph is gone only grad here
        grad_recover = torch.zeros_like(self.get_deform)
        grad_recover[:deform_verts_grad_backup.shape[0]] =  deform_verts_grad_backup
        self.get_deform.grad = grad_recover

        self.prune_alpha(min_alpha)
        self.clean_verts()

        torch.cuda.empty_cache()

    def add_densification_stats(self, update_filter,ndc_grad):
        # Camera Define in Screen Space

        # grad_ndc_verts = ndc_grad['verts_ndc']
        # grad_ndc_verts = grad_ndc_verts[0]
        # self.deform_verts_grad_accum[update_filter] += torch.norm(grad_ndc_verts[update_filter,:2], dim=-1, keepdim=True)

        self.deform_verts_grad_accum[update_filter] += torch.norm(self.get_deform.grad[update_filter], dim=-1, keepdim=True) 

                                                    #    torch.norm(self._alpha.grad[update_filter,None], dim=-1, keepdim=True)

        self.denom[update_filter] += 1
        
    def densify_and_split_flexible(self,mask, scene_extent):
        verts = self.get_verts
        faces = self.get_faces
        texts = self.get_textures('raw')

        selected_pts_mask = mask
        if selected_pts_mask.sum() == 0:
            return None,None,None

        extension_idx = [[0,1],[0,2],[1,2], [0,1],[0,2],[1,2] , [0,1],[0,2],[1,2]]
        new_faces = faces[selected_pts_mask][:,extension_idx]
        new_verts = verts[new_faces].mean(dim = -2)
        new_verts_idx = torch.arange(verts.shape[0],verts.shape[0] + new_verts.shape[0]*9 ,device='cuda').reshape(-1,9) # 3 4 5 6 7 8 9 10 11
        all_new_faces = torch.cat([faces[selected_pts_mask],new_verts_idx],dim=-1)

        combo = [[0,3,4],[2,5,7],[1,6,8],[9,10,11]]
        all_faces = all_new_faces[:,combo].reshape(-1,3)
        all_verts = torch.cat([verts,new_verts.reshape(-1,3)])
        all_faces = torch.cat([faces[~selected_pts_mask],all_faces] ,dim=0)
    
        new_deform_verts = torch.zeros_like(new_verts.reshape(-1,3)).requires_grad_(True)
        new_texts = self.deact(texts[:,faces[selected_pts_mask][:,extension_idx],:].mean(-2).requires_grad_(True).reshape(texts.shape[0],-1,texts.shape[-1]))
        new_ambient_materials = self.deact(self.get_ambient_materials[:,faces[selected_pts_mask][:,extension_idx],:].mean(-2).requires_grad_(True).reshape(self.get_ambient_materials.shape[0],-1,self.get_ambient_materials.shape[-1]))
        new_diffuse_materials =  self.deact(self.get_diffuse_materials[:,faces[selected_pts_mask][:,extension_idx],:].mean(-2).requires_grad_(True).reshape(self.get_diffuse_materials.shape[0],-1,self.get_diffuse_materials.shape[-1]))
        new_specular_materials =  self.deact(self.get_specular_materials[:,faces[selected_pts_mask][:,extension_idx],:].mean(-2).requires_grad_(True).reshape(self.get_specular_materials.shape[0],-1,self.get_specular_materials.shape[-1]))
        new_feature_dc_s = self._feature_dc_s[:,faces[selected_pts_mask][:,extension_idx],:].mean(-3).requires_grad_(True).reshape(self._feature_dc_s.shape[0],-1,self._feature_dc_s.shape[-2],self._feature_dc_s.shape[-1])
        new_feature_rest_s = self._feature_rest_s[:,faces[selected_pts_mask][:,extension_idx],:].mean(-3).requires_grad_(True).reshape(self._feature_rest_s.shape[0],-1,self._feature_rest_s.shape[-2],self._feature_rest_s.shape[-1])
        if self.materials_type =='CT':
            new_shininess =  self.deact(self.get_shininess[:,faces[selected_pts_mask][:,extension_idx],:].mean(-2).requires_grad_(True).reshape(self.get_shininess.shape[0],-1,self.get_shininess.shape[-1]))
        elif self.materials_type =='BP':
            new_shininess =  torch.log(self.get_shininess)[:,faces[selected_pts_mask][:,extension_idx],:].mean(-2).requires_grad_(True).reshape(self.get_shininess.shape[0],-1,self.get_shininess.shape[-1])
            new_shininess = torch.exp(new_shininess)
        new_alpha = self.deact(self.get_alpha[faces[selected_pts_mask][:,extension_idx]].mean(-1).requires_grad_(True).reshape(-1))

        print(f"Spliting {all_verts.shape[0]} New Verts and {all_faces.shape[0]} Faces")
        new_tensor_dict = {
        'deform':new_deform_verts,
        'textures':new_texts,
        'alpha':new_alpha,
        'ambient_materials': new_ambient_materials,
        'diffuse_materials': new_diffuse_materials,
        'specular_materials': new_specular_materials,
        'feature_dc_s':new_feature_dc_s,
        'feature_rest_s':new_feature_rest_s,
        'shininess': new_shininess
        }
        self.densification_postfix(new_tensor_dict,all_verts,all_faces)

    def sparsification_postfix(self, 
                              mask,
                              verts,
                              faces
                              ):
        
        optimizable_tensor = self._prune_optimizer(mask)
        self._deform_verts = optimizable_tensor['deform'].contiguous().requires_grad_(True)
        self._texts = optimizable_tensor['textures'].contiguous().requires_grad_(True)
        self._alpha = optimizable_tensor['alpha'].contiguous().requires_grad_(True)
        self._ambient_materials = optimizable_tensor['ambient_materials'].contiguous().requires_grad_(True)
        self._diffuse_materials = optimizable_tensor['diffuse_materials'].contiguous().requires_grad_(True)
        self._specular_materials = optimizable_tensor['specular_materials'].contiguous().requires_grad_(True)
        self._shininess = optimizable_tensor['shininess'].contiguous().requires_grad_(True)
        self._feature_dc_s = optimizable_tensor['feature_dc_s'].contiguous().requires_grad_(True)
        self._feature_rest_s = optimizable_tensor['feature_rest_s'].contiguous().requires_grad_(True)
        self.deform_verts_grad_accum = torch.zeros((self._deform_verts.shape[0], 1), device="cuda").contiguous()
        self.denom = torch.zeros((self._deform_verts.shape[0], 1), device="cuda").contiguous()
        self._src_mesh = Meshes(verts[mask].unsqueeze(0).contiguous(),faces.unsqueeze(0).contiguous())

    def densification_postfix(self, 
                              new_tensor_dict,
                              new_verts,
                              new_faces
                              ):
        
        optimizable_tensor = self.cat_tensors_to_optimizer(new_tensor_dict)
        self._deform_verts = optimizable_tensor['deform'].contiguous().requires_grad_(True)
        self._texts = optimizable_tensor['textures'].contiguous().requires_grad_(True)
        self._alpha = optimizable_tensor['alpha'].contiguous().requires_grad_(True)
        self._ambient_materials = optimizable_tensor['ambient_materials'].contiguous().requires_grad_(True)
        self._diffuse_materials = optimizable_tensor['diffuse_materials'].contiguous().requires_grad_(True)
        self._specular_materials = optimizable_tensor['specular_materials'].contiguous().requires_grad_(True)
        self._shininess = optimizable_tensor['shininess'].contiguous().requires_grad_(True)
        self._feature_dc_s = optimizable_tensor['feature_dc_s'].contiguous().requires_grad_(True)
        self._feature_rest_s = optimizable_tensor['feature_rest_s'].contiguous().requires_grad_(True)
        self.deform_verts_grad_accum = torch.zeros((self._deform_verts.shape[1], 1), device="cuda").contiguous()
        self.denom = torch.zeros((self._deform_verts.shape[1], 1), device="cuda").contiguous()

        new_src_mesh = Meshes(new_verts.unsqueeze(0).contiguous(),new_faces.unsqueeze(0).contiguous())
        new_src_mesh.textures = TexturesVertex(verts_features=self._texts.contiguous()) 
        self._src_mesh = new_src_mesh.offset_verts(-self._deform_verts)

        new_area = torch.zeros(self.get_faces.shape[0] - self.max_screen_area.shape[0]).to('cuda').contiguous()
        self.max_screen_area = torch.cat([self.max_screen_area,new_area]).contiguous()

    def densify_and_clone(self, mask, scene_extent):
        verts = self.get_verts
        faces = self.get_faces
        texts = self.get_textures(texts_format = 'raw')
  
        selected_pts_mask = mask
        if selected_pts_mask.sum() == 0:
            return
        new_faces = faces[selected_pts_mask]
        new_verts = verts[new_faces].detach().clone()

        new_verts_idx = torch.arange(verts.shape[0],verts.shape[0] + new_verts.shape[0]*3 ,device='cuda').reshape(-1,3) # 3 4 5 
        all_new_faces = torch.cat([faces[selected_pts_mask],new_verts_idx],dim=-1)

        combo = [[3,4,5]]
        new_faces = all_new_faces[:,combo].reshape(-1,3)
        all_verts = torch.cat([verts,new_verts.reshape(-1,3)])
        all_faces = torch.cat([faces,new_faces] ,dim=0)

        new_deform_verts = torch.zeros_like(new_verts.reshape(-1,3)).requires_grad_(True)
        new_texts = self.deact(texts[:,faces[selected_pts_mask],:].requires_grad_(True).reshape(texts.shape[0],-1,texts.shape[-1]))
        new_ambient_materials = self.deact(self.get_ambient_materials[:,faces[selected_pts_mask],:].requires_grad_(True).reshape(self.get_ambient_materials.shape[0],-1,self.get_ambient_materials.shape[-1]))
        new_diffuse_materials = self.deact(self.get_diffuse_materials[:,faces[selected_pts_mask],:].requires_grad_(True).reshape(self.get_diffuse_materials.shape[0],-1,self.get_diffuse_materials.shape[-1]))
        new_specular_materials = self.deact(self.get_specular_materials[:,faces[selected_pts_mask],:].requires_grad_(True).reshape(self.get_specular_materials.shape[0],-1,self.get_specular_materials.shape[-1]))
        if self.materials_type=='CT':
            new_shininess = self.get_shininess[:,faces[selected_pts_mask],:].requires_grad_(True).reshape(self.get_shininess.shape[0],-1,self.get_shininess.shape[-1])
            new_shininess = self.deact(new_shininess)
        elif self.materials_type == 'BP':
            new_shininess = torch.log(self.get_shininess)[:,faces[selected_pts_mask],:].requires_grad_(True).reshape(self.get_shininess.shape[0],-1,self.get_shininess.shape[-1])
            new_shininess = torch.exp(self.deact(new_shininess))
        new_alpha = self.deact(self.get_alpha[faces[selected_pts_mask]].requires_grad_(True).reshape(-1))
        new_feature_dc_s = self._feature_dc_s[:,faces[selected_pts_mask],...].requires_grad_(True).reshape(self._feature_dc_s.shape[0],-1,self._feature_dc_s.shape[-2],self._feature_dc_s.shape[-1])
        new_feature_rest_s = self._feature_rest_s[:,faces[selected_pts_mask],...].requires_grad_(True).reshape(self._feature_rest_s.shape[0],-1,self._feature_rest_s.shape[-2],self._feature_rest_s.shape[-1])

        print(f"Cloning {all_verts.shape[0]} New Verts and {all_faces.shape[0]} Faces")
        new_tensor_dict = {
        'deform':new_deform_verts,
        'textures':new_texts,
        'alpha':new_alpha,
        'ambient_materials': new_ambient_materials,
        'diffuse_materials': new_diffuse_materials,
        'specular_materials': new_specular_materials,
        'shininess': new_shininess,
        'feature_dc_s':new_feature_dc_s,
        'feature_rest_s':new_feature_rest_s,
        }
        self.densification_postfix(new_tensor_dict,all_verts,all_faces)

    def _prune_optimizer(self,mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if 'light' in group['name']:
                continue
            if stored_state is not None:
                if group['name'] == 'deform':
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask,...]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask,...]
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask,...].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                elif group['name'] == 'alpha':
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask,...]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask,...]
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask,...].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    stored_state["exp_avg"] = stored_state["exp_avg"][:,mask,...]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][:,mask,...]
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][:,mask,...].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group['name'] == 'deform':
                    group["params"][0] = nn.Parameter(group["params"][0][mask,...].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
                elif group['name'] == 'alpha' :
                    group["params"][0] = nn.Parameter(group["params"][0][mask,...].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][:,mask,...].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    def cat_tensors_to_optimizer(self,tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "light" in group["name"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if len(extension_tensor.shape)==2 or len(extension_tensor.shape)==1:
                dim_cat = 0
            else:
                dim_cat = 1
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=dim_cat)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=dim_cat)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=dim_cat).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=dim_cat).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    def clean_verts(self):
        faces = self._src_mesh.faces_packed()
        verts = self._src_mesh.verts_packed()
        mask = torch.zeros(verts.shape[0],dtype=torch.bool,device='cuda')
        mask[faces.reshape(-1).unique()] = True
        # Sweep faces, Old index to new index 
        remaining_indices = torch.nonzero(mask).squeeze(-1)
        old_to_new_index_map = torch.full((verts.shape[0],), -1, dtype=torch.int64,device='cuda')
        old_to_new_index_map[remaining_indices] = torch.arange(len(remaining_indices),device='cuda')
        new_faces = torch.gather(old_to_new_index_map[faces], 1, torch.arange(faces.shape[1],device=faces.device).expand(faces.shape[0], faces.shape[1])) # TODO: Make a CUDA to do this 
        
        self.sparsification_postfix(mask,verts,new_faces)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor) 
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def reset_alpha(self):
        alpha = self.get_alpha
        opacities_new = inverse_sigmoid(torch.min(alpha, torch.ones_like(alpha)*self.alpha_reset_val))

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "alpha")

        self._alpha = optimizable_tensors["alpha"]

    def prune_large_mesh_world(self,scene_extent):
        area, normal = mesh_face_areas_normals(self.get_verts,self.get_faces) 
        big_mesh_ws = area > 0.1 * scene_extent
        big_mesh_mask = (~big_mesh_ws)

        faces = self.get_faces

        idx = faces[big_mesh_mask].unsqueeze(0)
        self._src_mesh = Meshes(self._src_mesh.verts_packed().unsqueeze(0),idx)
        self.max_screen_area = self.max_screen_area[big_mesh_mask]

    def prune_alpha(self,min_alpha):
        faces = self.get_faces
        centroid_alphas = self.get_alpha[faces].max(dim=1)[0]
        minus_mask = centroid_alphas <= (min_alpha)
        print(f"Pruning Alpha: {minus_mask.sum()},Avg_Centroid_Alpha: {centroid_alphas.mean()}")
        alpha_mask = (~minus_mask).squeeze(-1)
        idx = faces[alpha_mask].unsqueeze(0)
        self._src_mesh = Meshes(self._src_mesh.verts_packed().unsqueeze(0),idx)
        self.max_screen_area = self.max_screen_area[alpha_mask]

    def subdivide_mesh_loop(self):
        # self.new_src_mesh = src_mesh.offset_verts(deform_verts)
        mesh = self.get_mesh
        V = mesh.num_verts_per_mesh()[0]
        deform_vert = self.get_deform
        texts = self.get_textures('raw')
        alpha = self.get_alpha
        ambient_materials = self.get_ambient_materials
        diffuse_materials = self.get_diffuse_materials
        specular_materials = self.get_specular_materials
        emission_materials = self.get_emission_materials
        shininess = self.get_shininess
        # texts = mesh.textures.faces_verts_textures_packed()

        new_mesh,new_deform_vert = SubdivideMeshes()(mesh.clone(),deform_vert)

        _,texts = SubdivideMeshes()(mesh.clone(),texts[0])
        _,new_alpha = SubdivideMeshes()(mesh.clone(),alpha)


        _,ambient_materials = SubdivideMeshes()(mesh.clone(),ambient_materials[0])
        _,diffuse_materials = SubdivideMeshes()(mesh.clone(),diffuse_materials[0])
        _,specular_materials = SubdivideMeshes()(mesh.clone(),specular_materials[0])
        _,emission_materials = SubdivideMeshes()(mesh.clone(),emission_materials[0])
        _,shininess = SubdivideMeshes()(mesh.clone(),shininess[0])

        print('Subdivide Mesh, now vert num: ', new_mesh.verts_packed().shape[0])
        new_tensor_dict = {
            'deform':new_deform_vert[V:],
            'textures': texts[V:].unsqueeze(0),
            'alpha':new_alpha[V:],
            'ambient_materials':ambient_materials[V:].unsqueeze(0),
            'diffuse_materials': diffuse_materials[V:].unsqueeze(0),
            'specular_materials': specular_materials[V:].unsqueeze(0),
            'shininess':shininess[V:].unsqueeze(0)
            }
        self.densification_postfix(new_tensor_dict,new_mesh.verts_packed(),new_mesh.faces_packed())

    def check_progress(self,iteration,opt):
        # if iteration < 1000:
        #     self._shininess.requires_grad = False
        #     self._specular_materials.requires_grad = False
        #     self._ambient_materials.requires_grad = False
        # else:
        #     self._shininess.requires_grad = True
        #     self._specular_materials.requires_grad = True
        #     self._ambient_materials.requires_grad = True
        if 1000>=iteration >0:
            self.sh_degree = 0
        elif 2000 >=iteration>1000:
            self.sh_degree = 1
        elif 3000 >= iteration > 2000:
            self.sh_degree = 2
        elif 7000 >= iteration > 3000:
            self.sh_degree = 3
        elif 15000 >= iteration >7000:
            self.sh_degree = 4
        else :
            self.sh_degree = 5
        # Faster, 
        if iteration < 200:
            self.resolution_scale = 1/4
            self.faces_per_pixel = 150 #40
            # self.percent_dense = 1e-2
            # self.max_grad = 1e-4
            # self.min_alpha = 0.01
        elif 600>iteration >=200:
            self.resolution_scale = 1/2
            self.faces_per_pixel = opt.faces_per_pixel # 30
            # self.percent_dense = 1e-2
            # self.min_alpha = 0.05 # Make the surface slim  / 25
            # self.alpha_reset_val = 0.01
            # self.max_grad = 1e-4
        elif 10000>iteration >=600:
            self.resolution_scale = 1.
            self.faces_per_pixel = opt.faces_per_pixel # 30
            # self.percent_dense = 1e-2
            # self.min_alpha = 0.05 # Make the surface slim  / 25
            # self.alpha_reset_val = 0.01
            # self.max_grad = 1e-4
        elif 20000>iteration >=10000:
            self.resolution_scale = 1    
            self.faces_per_pixel = opt.faces_per_pixel
            # self.percent_dense = 1e-2
            # self.min_alpha = 0.05 # Make the surface slim
            # self.alpha_reset_val = 0.01
            # self.max_grad = 1e-4
        else:
            self.resolution_scale = 1    
            self.faces_per_pixel = opt.faces_per_pixel
            # self.percent_dense = 1e-2
            self.min_alpha = 0.05 # Make the surface slim
            # self.alpha_reset_val = 0.01
            # self.max_grad = 1e-4