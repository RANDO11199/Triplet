def align_paramater_relfectionmodel(LightArgs,optArgs,modelArgs):
    if modelArgs.materials_type =="BP":
        LightArgs.light_ambient_color = (1.,1.,1.)
        LightArgs.light_ambient_lr = 0.005
        optArgs.shininess_lr = 0.001
    elif modelArgs.materials_type =="CT":
        LightArgs.light_ambient_color = (0.003,0.003,0.003)
        LightArgs.light_ambient_lr = 0.0001
        optArgs.shininess_lr = 0.01
    return LightArgs,optArgs
    
def align_paramater_scene_type(LightArgs,optArgs,modelArgs):
    if optArgs.scene_type == "Scene":
        LightArgs.max_sh_degree = 2
        LightArgs.compensate_random_Point=True
        optArgs.random_background =True
        optArgs.deform_lr_init = 0.00016
        optArgs.deform_lr_final = 0.0000016
        optArgs.iterations = 30000
        optArgs.densify_grad_threshold = 7.5e-5
        modelArgs._white_background = False
    elif optArgs.scene_type=='Object':
        LightArgs.max_sh_degree = 5
        LightArgs.compensate_random_Point=False
        optArgs.random_background =False
        optArgs.deform_lr_init = 0.00011
        optArgs.deform_lr_final = 0.0000011
        optArgs.iterations = 18000
        optArgs.densify_grad_threshold = 2e-4
    else:
        raise NotImplementedError(f'Scene type : {optArgs.scene_type} not implemented yet')
    return LightArgs,optArgs