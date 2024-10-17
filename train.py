#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# For inquiries contact  george.drettakis@inria.fr

# Copyright (C) 2024, Jiajie Yang https://github.com/RANDO11199
# All rights reserved.

# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
#
import warnings
warnings.filterwarnings('ignore')
from torchvision.utils import save_image,make_grid
import os
import torch
from pytorch3d.io import save_ply
from random import randint
from renderer import Rasterize
from utils.loss_utils import l1_loss, ssim,tv_loss
from pytorch3d.loss import mesh_edge_loss,mesh_laplacian_smoothing,mesh_normal_consistency
from renderer import SHrender,network_gui,RayTracer
from renderer.render_utils import Init_shaer
import sys
from scene import Scene,TripletModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from torch.nn.functional import interpolate
from utils.loss_utils import psnr
from argparse import ArgumentParser, Namespace
from arguements import ModelParams, PipelineParams, OptimizationParams, LightParams, align_paramater_relfectionmodel, align_paramater_scene_type
from functools import partial
torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, lightset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    triplet = TripletModel()
    scene = Scene(dataset, lightset, triplet)
    triplet.training_setup(opt)
    shader = Init_shaer(dataset)
    if checkpoint:
        (model_params,scene_params, first_iter) = torch.load(checkpoint)
        triplet.restore(model_params, opt)
        scene.restore(scene_params,lightset)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    if dataset.renderer == 'SH':
        render = SHrender
    elif dataset.renderer == 'Rasterization':
        render = partial(Rasterize,shader)
    elif dataset.renderer == 'Whitted-StyleRT':
        render = RayTracer
    else:
        raise NotImplementedError(f'Renderer {dataset.renderer} is not implemented yet')

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        triplet.check_progress(iteration,opt)
        scene.check_progress(iteration)
        # TODO: Currently not supported
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(shader,custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        iter_start.record()
        triplet.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        rand_num = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(rand_num)
        light,diffuse_map,specular_map = scene.getLightSource()
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        # with torch.autograd.profiler.profile(enabled=True,use_cuda=True,with_flops=True,profile_memory=True,use_cpu=False,use_kineto=True) as prof:
        render_pkg = render(target_camera = viewpoint_cam, light = light, triplet = triplet, pipe = pipe, bg_color = bg,scaling=triplet.resolution_scale)
        # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
        image, visibility_filter, area, depth_map, normal_map,ndc_grad = render_pkg["render"],  render_pkg["visibility_filter"], render_pkg["area"],render_pkg['depth_map'],render_pkg['normal_map'],render_pkg['ndc_grad']

        # Loss
        gt_image = viewpoint_cam.original_image[None]
        gt_image = interpolate(gt_image,scale_factor=triplet.resolution_scale)[0]
        gt_mask  = viewpoint_cam.gt_alpha_mask
        if gt_mask is not None : 
            gt_mask = interpolate(gt_mask[None],scale_factor=triplet.resolution_scale)[0]
            gt_image = gt_image + torch.ones_like(gt_image,device=gt_image.device) * bg.reshape(3,1,1) * ( 1- gt_mask)
        Ll1 = l1_loss(image, gt_image,reduce='mean') 
        loss = (1.0 - opt.lambda_dssim) * (Ll1 ) + opt.lambda_dssim *(1.0 - ssim(image, gt_image))
        if iteration >= 7000:
            loss += (0.01 * tv_loss(image[None]) +
                     0.05 * tv_loss(render_pkg['pseudo_normal'][None])
                     ) # Local Invariants/Continuous
            loss += 0.05 * l1_loss(render_pkg['pseudo_normal'][None],normal_map,reduce='mean')

        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(triplet._feature_rest_s,1.)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),
                            render_img=image,gt_image = gt_image,normal_image =render_pkg['normal_map'],depth_image=depth_map,opt=dataset,pseudo_normal = render_pkg['pseudo_normal'])
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Triplet".format(iteration))
                scene.save(iteration)
            triplet.max_screen_area[visibility_filter[0]] = torch.max(triplet.max_screen_area[visibility_filter[0]],area)
            # Kill large mesh on screen
            if iteration < opt.densify_from_iter or iteration % opt.densification_interval == 0:
                triplet.kill_large_mesh(max_screen_size = gt_image.shape[1]*gt_image.shape[2] * opt.screen_max )
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                triplet.add_densification_stats(visibility_filter[1],ndc_grad)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:

                    triplet.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    triplet.reset_alpha()
            # Optimizer step
            if iteration < opt.iterations:
                triplet.optimizer.step()
                triplet.optimizer.zero_grad(set_to_none=True)
                scene.optimizer.step()
                scene.optimizer.zero_grad(set_to_none=True)
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((triplet.capture(),scene.capture() ,iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            # triplet.setup_view_dependent_compensation(iteration,opt.view_compensation_from_iter)
            # scene.check_grad_progress(iteration,opt)
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer:SummaryWriter, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,**kwargs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        if  iteration % 100 == 0 :
            # scene.triplets.save_ply_geometry(iteration)
            try:
                sav_img = torch.stack([kwargs['render_img'],kwargs['gt_image'],
                                        kwargs['normal_image'][0],(kwargs['pseudo_normal']),
                                        torch.from_numpy(kwargs['depth_image']).permute(2,0,1).to('cuda')
                                        ],dim =0)
                sav_img = make_grid(sav_img,nrow=2)
                save_image(sav_img,f'./output/test_{iteration}.png')
            except:
                print('saving image fail')
                
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    lightsource,_,_ = scene.getLightSource()
                    result = renderFunc(target_camera = viewpoint, light = lightsource,bg_color = renderArgs[1],pipe=None,triplet= scene.triplets)["render"]
                    image = torch.clamp(result, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0) 
                    if viewpoint.gt_alpha_mask is not None:
                        gt_image = gt_image + torch.ones_like(gt_image).cuda()  * ( 1-  viewpoint.gt_alpha_mask)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.triplets.get_alpha, iteration)
            tb_writer.add_scalar('total_points', scene.triplets.get_deform.shape[1], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    lp = LightParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6019)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 14000,16000, 18000,19000,20000, 21000,22000,23000,24000,25000,30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])# Come back later
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--align_param', action='store_true', default=True)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(lp.compensate_random_Point)
    lp = lp.extract(args)
    mp = mp.extract(args)
    op = op.extract(args)
    if args.align_param:
        lp,op = align_paramater_relfectionmodel(lp,op,mp)
        lp,op = align_paramater_scene_type(lp,op,mp)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port) 
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(mp,lp, op, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
