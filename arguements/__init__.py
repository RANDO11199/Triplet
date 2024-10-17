from argparse import ArgumentParser, Namespace
import sys
import os
from .align_config import align_paramater_relfectionmodel,align_paramater_scene_type
class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = True
        self.materials_type = "BP" # BP SH RT
        self.renderer = 'Rasterization'
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class LightParams(ParamGroup):
    def __init__(self, parser):
        self.LStype = 'SH' # Point/Direction/EnvMap/SH
        self.light_intensity = ( 40.,40.,40) # If None, use Square of Scene Radius
        self.ambient_intensity = (40.,40.,40.)
        self.light_specular_color =  (1.,1.,1.)
        self.light_diffuse_color =   (1.,1.,1.)
        self.light_ambient_color = (1.,1.,1.) # (1.,1.,1.) for BlinnPhong (0.003,0.003,0.003)  for CookTorrance
        self.light_position_lr = 0.005
        self.light_direction_lr = 0.05
        self.light_ambient_lr = 0.005 # 0.0001 fir CT 0.005 for BP
        self.light_diffuse_lr = 0.005
        self.light_specular_lr = 0.005
        self.light_intensity_lr = 0.5
        self.light_ambient_intensity_lr = 0.05
        self.sh_band = 50
        self.diffuse_band = 121
        self.specular_band = 9
        self.envmap_resolution = (800,800) # W/phi yawing 2pi H/theta pitch pi
        self.diffuse_sh_coefs_lr = 0.02 # EnvMap
        self.specular_sh_coefs_lr = 0.02 # EnvMap
        self.max_sh_degree = 5 # scene 2 object 5
        self.compensate_random_Point = True # True for scene
        super().__init__(parser, "Optimization Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.scene_type = 'Object' # Scene, Object
        self.iterations = 18000 # 18000 for object 30000 for scene
        self.deform_lr_init = 0.00011 # 0.00016 for  scene 0.00011 for object
        self.deform_lr_final = 0.0000011 # 0.0000016 for  scene  0000011 for object
        self.lr_delay_steps = 0.
        self.deform_lr_delay_mult = 1.
        self.deform_lr_max_steps = 30_000
        self.SH_lr = 0.001
        self.ambient_materials_lr = 0.001
        self.diffuse_materials_lr = 0.001
        self.specular_materials_lr = 0.001
        self.shininess_lr = 0.001 # 0.001 for blinnphong 0.01 for cooktorrance
        self.texts_lr = 0.001
        self.alpha_lr = 0.05
        self.deform_lr = 0.00016 
        self.percent_dense = 0.01
        self.screen_max = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 7.5e-5 # 7.5e-5 , 1e-4  if no engouh vram
        self.complex_materials_start_from = 0
        self.random_background = False
        self.use_gt_alpha = False
        self.faces_per_pixel = 20 # 15 for object 10 for scene
                                    # For semi-transparent and transparent object, set higher faces_per_pixel and less alpha; You could also set high faces_per_pixel and low alpha directly, but it would increase the cost (i.e. H*W*FacesPerPixel, double FPP, double cost)
        # self.view_compensation_from_iter = 30000
        # self.improve_density_from_iter = [1000,3000,5000]
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
