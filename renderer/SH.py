import torch
from torch.autograd import Function
torch.set_default_device('cuda')
from einops import einsum
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792,-1.0925484305920792,0.31539156525252005,-1.0925484305920792,0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435)

def RGB2SH(rgb):
    return (rgb - 0.5) / SH_C0

def SH2RGB(sh):
    return sh * SH_C0 + 0.5
def dnormvdv3d(unnormalized_v:torch.Tensor,
                direction_v_x : torch.Tensor,
                direction_v_y : torch.Tensor,
                direction_v_z : torch.Tensor):
    sum2 = unnormalized_v[...,0] * unnormalized_v[...,0] + unnormalized_v[...,1] * unnormalized_v[...,1]+\
           unnormalized_v[...,2] * unnormalized_v[...,2] # (batch, vert,1)
    invsum32 = 1.0/torch.sqrt(sum2 * sum2 * sum2)

    dnormvdv = torch.zeros_like(unnormalized_v)
    dnormvdv[...,0] = ((+sum2 - unnormalized_v[...,0] * unnormalized_v[...,0]) * direction_v_x
                       - unnormalized_v[...,1] * unnormalized_v[...,0] * direction_v_y 
                       - unnormalized_v[...,2] * unnormalized_v[...,0] * direction_v_z) * invsum32
    
    dnormvdv[...,1] = (-unnormalized_v[...,0] * unnormalized_v[...,1] * direction_v_x + 
                       (sum2 - unnormalized_v[...,1] * unnormalized_v[...,1]) * direction_v_y -
                        unnormalized_v[...,2] * unnormalized_v[...,1] * direction_v_z ) * invsum32
    
    dnormvdv[...,2] = (-unnormalized_v[...,0] * unnormalized_v[...,2] * direction_v_x -
                       unnormalized_v[...,1] * unnormalized_v[...,2] * direction_v_y + 
                       (sum2 - unnormalized_v[...,2] * unnormalized_v[...,2]) * direction_v_z)*invsum32
    return dnormvdv

class SphereHarmonic(Function):
    @staticmethod
    def forward(ctx,
                position:torch.Tensor,
                degree:int,
                max_coeffs:int,
                camera_positions:torch.FloatTensor,
                SphereHarmonics_Coeffs:torch.FloatTensor, # (batch size/mesh, vertex,rgb, max_coeffs)
                )-> torch.Tensor:
        
        # position = mesh.verts_packed() # (Batch Size/ Mesh, Number of Vertexes, xyz)
        dir = position - camera_positions # (Batch Size/ Mesh, Number of Vertexes, xyz)
        dir = dir / dir.norm(p=2,dim=-1).unsqueeze(dim = -1) 
        result = SH_C0 * SphereHarmonics_Coeffs[...,0] #0阶 方向无关，Diffuse
        if (degree>0):
            x = dir[...,0].unsqueeze(dim=-1)
            y = dir[...,1].unsqueeze(dim=-1)
            z = dir[...,2].unsqueeze(dim=-1)
            result = result - \
                              SH_C1 * y * SphereHarmonics_Coeffs[...,1] +\
                              SH_C1 * z * SphereHarmonics_Coeffs[...,2] -\
                              SH_C1 * x * SphereHarmonics_Coeffs[...,3]
            if (degree > 1):
                xx = x*x
                yy = y*y
                zz = z*z
                xy = x*y
                yz = y*z
                xz = x*z
                result = result + \
                                SH_C2[0] * xy * SphereHarmonics_Coeffs[...,4] + \
                                SH_C2[1] * yz * SphereHarmonics_Coeffs[...,5] + \
                                SH_C2[2] * (2.0 * zz - xx - yy) * SphereHarmonics_Coeffs[...,6] + \
                                SH_C2[3] * xz * SphereHarmonics_Coeffs[...,7] + \
                                SH_C2[4] * (xx - yy) * SphereHarmonics_Coeffs[...,8]
                if (degree > 2):
                    result = result + \
                                      SH_C3[0] * y * (3.0 * xx - yy) * SphereHarmonics_Coeffs[...,9] + \
                                      SH_C3[1] * xy * z *SphereHarmonics_Coeffs[...,10] + \
                                      SH_C3[2] * y * (4.0 * zz - xx - yy) * SphereHarmonics_Coeffs[...,11] + \
                                      SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * SphereHarmonics_Coeffs[...,12] +\
                                      SH_C3[4] * x * (4.0 * zz - xx - yy) * SphereHarmonics_Coeffs[...,13] + \
                                      SH_C3[5] * z * (xx - yy) * SphereHarmonics_Coeffs[...,14] + \
                                      SH_C3[6] * x * (xx - 3.0 * yy) * SphereHarmonics_Coeffs[...,15]
        result += 0.5
        ctx.sh_config = (max_coeffs,degree)
        clamped = result < 0
        ctx.save_for_backward(position, SphereHarmonics_Coeffs, camera_positions, clamped)

        return torch.nn.functional.relu(result)
    
    @staticmethod
    def backward(ctx,dL_dcolor):
        
        position, SphereHarmonics_Coeffs, camera_positions, clamped = ctx.saved_tensors 
        position = position
        SphereHarmonics_Coeffs = SphereHarmonics_Coeffs
        camera_positions = camera_positions
        clamped = clamped
        max_coeffs, degree = ctx.sh_config

        dir_orig = position - camera_positions 
        dir_orig = dir_orig

        dir = dir_orig / dir_orig.norm(dim=2).unsqueeze(dim = -1)
        x = dir[...,0].unsqueeze(dim=-1)
        y = dir[...,1].unsqueeze(dim=-1)
        z = dir[...,2].unsqueeze(dim=-1)

        dL_dRGB = dL_dcolor# (Batch size , vertex, RGB)

        mask_clamped = torch.where(clamped, 0, 1)
        
        dL_dRGB = dL_dRGB * mask_clamped

        dRGBdsh0 = SH_C0

        dL_dsh = torch.zeros( dL_dRGB.shape[0], dL_dRGB.shape[1], 3 , max_coeffs,device='cuda')

        dL_dsh[...,0] = dL_dRGB * dRGBdsh0
        dRGBdx = torch.zeros(SphereHarmonics_Coeffs.shape[0],SphereHarmonics_Coeffs.shape[1],3,1,device='cuda')
        dRGBdy = torch.zeros(SphereHarmonics_Coeffs.shape[0],SphereHarmonics_Coeffs.shape[1],3,1,device='cuda')
        dRGBdz = torch.zeros(SphereHarmonics_Coeffs.shape[0],SphereHarmonics_Coeffs.shape[1],3,1,device='cuda')

        if (degree > 0):
            dRGBdsh1 = (-SH_C1 * y)
            dRGBdsh2 = (SH_C1 * z)
            dRGBdsh3 = (-SH_C1 * x)
            dL_dsh[...,1] = dRGBdsh1 * dL_dRGB
            dL_dsh[...,2] = dRGBdsh2 * dL_dRGB
            dL_dsh[...,3] = dRGBdsh3 * dL_dRGB

            dRGBdx = -SH_C1 * SphereHarmonics_Coeffs[...,3]
            dRGBdy = -SH_C1 * SphereHarmonics_Coeffs[...,1]
            dRGBdz = SH_C1 * SphereHarmonics_Coeffs[...,2]

            if degree > 1:
                xx = x * x
                yy = y * y
                zz = z * z 
                xy = x * y 
                yz = y * z 
                xz = x * z 
                
                dRGBdsh4 = (SH_C2[0] * xy )
                dRGBdsh5 = (SH_C2[1] * yz)
                dRGBdsh6 = (SH_C2[2] * (2.0 * zz - xx - yy))
                dRGBdsh7 = (SH_C2[3] * xz)
                dRGBdsh8 = (SH_C2[4] * (xx - yy))
                
                dL_dsh[...,4] = dRGBdsh4 * dL_dRGB
                dL_dsh[...,5] = dRGBdsh5 * dL_dRGB
                dL_dsh[...,6] = dRGBdsh6 * dL_dRGB
                dL_dsh[...,7] = dRGBdsh7 * dL_dRGB
                dL_dsh[...,8] = dRGBdsh8 * dL_dRGB
                
                dRGBdx += SH_C2[0] * y * SphereHarmonics_Coeffs[...,4] +  \
                        SH_C2[2] * 2. * -x * SphereHarmonics_Coeffs[...,6] + \
                        SH_C2[3] * z * SphereHarmonics_Coeffs[...,7] + \
                        SH_C2[4] * 2.0 * x * SphereHarmonics_Coeffs[...,8]
                        
                dRGBdy += SH_C2[0] * x * SphereHarmonics_Coeffs[...,4] + \
                        SH_C2[1] * z * SphereHarmonics_Coeffs[...,5] + \
                        SH_C2[2] * 2.0 * -y * SphereHarmonics_Coeffs[...,6] + \
                        SH_C2[4] * 2.0 * -y *SphereHarmonics_Coeffs[...,8]
                
                dRGBdz += SH_C2[1] * y *SphereHarmonics_Coeffs[...,5] + \
                        SH_C2[2] * 2.0 * 2.0 * z * SphereHarmonics_Coeffs[...,6] + \
                        SH_C2[3] * x * SphereHarmonics_Coeffs[...,7]
                
                if degree >2 :
                    dRGBdsh9 = SH_C3[0] * y * (3.0 * xx - yy)
                    dRGBdsh10 = SH_C3[1] * xy * z 
                    dRGBdsh11 = SH_C3[2] * y * (4.0 * zz - xx - yy)
                    dRGBdsh12 = SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                    dRGBdsh13 = SH_C3[4] * x * (4.0 * zz - xx - yy)
                    dRGBdsh14 = SH_C3[5] * z * (xx - yy)
                    dRGBdsh15 = SH_C3[6] * x * (xx - 3.0 *yy)
                    dL_dsh[...,9] = dRGBdsh9 * dL_dRGB
                    dL_dsh[...,10] = dRGBdsh10 * dL_dRGB
                    dL_dsh[...,11] = dRGBdsh11 * dL_dRGB
                    dL_dsh[...,12] = dRGBdsh12 * dL_dRGB
                    dL_dsh[...,13] = dRGBdsh13 * dL_dRGB
                    dL_dsh[...,14] = dRGBdsh14 * dL_dRGB
                    dL_dsh[...,15] = dRGBdsh15 * dL_dRGB

                    dRGBdx += SH_C3[0] * SphereHarmonics_Coeffs[...,9] * 3.0 * 2.0 * xy +\
                            SH_C3[1] * SphereHarmonics_Coeffs[...,10] * yz + \
                            SH_C3[2] * SphereHarmonics_Coeffs[...,11] * -2.0 * xy + \
                            SH_C3[3] * SphereHarmonics_Coeffs[...,12] * -3.0 * 2.0 * xz + \
                            SH_C3[4] * SphereHarmonics_Coeffs[...,13] * (-3.0 * xx + 4.0 * zz - yy) + \
                            SH_C3[5] * SphereHarmonics_Coeffs[...,14] * 2.0 * xz + \
                            SH_C3[6] * SphereHarmonics_Coeffs[...,15] * 3.0 * (xx - yy)
                            
                    
                    dRGBdy += SH_C3[0] * SphereHarmonics_Coeffs[...,9] * 3.0 * (xx - yy) +\
                            SH_C3[1] * SphereHarmonics_Coeffs[...,10] * xz + \
                            SH_C3[2] * SphereHarmonics_Coeffs[...,11] * (-3.0 * yy + 4.0 * zz - xx) +\
                            SH_C3[3] * SphereHarmonics_Coeffs[...,12] * -3.0 * 2.0 * yz +\
                            SH_C3[4] * SphereHarmonics_Coeffs[...,13] * -2.0 * xy + \
                            SH_C3[5] * SphereHarmonics_Coeffs[...,14] * -2.0 * yz + \
                            SH_C3[6] * SphereHarmonics_Coeffs[...,15] * -3.0 * 2.0 * xy
                    
                    
                    dRGBdz += SH_C3[1] * SphereHarmonics_Coeffs[...,10] * xy + \
                            SH_C3[2] * SphereHarmonics_Coeffs[...,11] * 4.0 * 2.0 * yz +\
                            SH_C3[3] * SphereHarmonics_Coeffs[...,12] * 3.0 * (2.0 * zz - xx - yy) +\
                            SH_C3[4] * SphereHarmonics_Coeffs[...,13] * 4.0 * 2.0 * xz + \
                            SH_C3[5] * SphereHarmonics_Coeffs[...,14] * (xx - yy)
                    

        dL_dposition = dnormvdv3d(dir_orig,
                              direction_v_x= (dRGBdx.squeeze(dim=-1) * dL_dRGB).sum(dim=-1),
                              direction_v_y = (dRGBdy.squeeze(dim=-1) * dL_dRGB).sum(dim=-1),
                              direction_v_z= (dRGBdz.squeeze(dim=-1) * dL_dRGB).sum(dim=-1))

        # print('backward position',dL_dpositions)

        return dL_dposition, None, None, None, dL_dsh
    

if __name__ == "__main__":
    from torch.autograd import gradcheck
    position = torch.randn(1,5,3,requires_grad=True).double().cuda()
    degree = 3
    max_coeffs = 16
    camera_positions = torch.tensor([0.,0.,-0.]).double().cuda()
    SphereHarmonics_Coeffs = torch.randn(1,5,3,max_coeffs,requires_grad=True).double().cuda()

    test = gradcheck(SphereHarmonic.apply,(position,degree,max_coeffs,camera_positions,SphereHarmonics_Coeffs),
                     eps=1e-6,atol=1e-6,raise_exception=True)
    print('111',test)
