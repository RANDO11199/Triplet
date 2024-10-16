# Inspired by and partially adapted from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/dataset_readers.py
# Very Good Note : https://medium.com/maochinn/%E7%AD%86%E8%A8%98-camera-dee562610e71
# https://amytabb.com/tips/tutorials/2019/06/28/OpenCV-to-OpenGL-tutorial-essentials/
# https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/row-major-vs-column-major-vector.html?source=post_page-----dee562610e71--------------------------------
from .colmap_utils import *
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from PIL import Image as PIL_IMAGE
from plyfile import PlyData, PlyElement
import json
from utils.sh_utils import SH2RGB
from pathlib import Path
CAM_SIMPLE_PINHOLE = collections.namedtuple(
    "CAM_SIMPLE_PINHOLE", ['f','cx','cy','model','fovx','fovy'])
CAM_PINHOLE = collections.namedtuple(
    "CAM_PINHOLE", ['fx','fy','cx','cy','model','fovx','fovy'])
CAM_SIMPLE_RADIAL = collections.namedtuple(
    "CAM_SIMPLE_RADIAL", ['f','cx','cy','k','model','fovx','fovy'])
CAM_RADIAL = collections.namedtuple(
    "CAM_RADIAL", ['f','cx','cy','k1','k2','model','fovx','fovy'])
CAM_OPENCV  = collections.namedtuple(
    "CAM_OPENCV", ['fx','fy','cx','cy','k1','k2','p1','p2','model','fovx','fovy'])

def parse_camera_model(cam_model):
    if cam_model.model =='SIMPLE_PINHOLE':
        intrinsics = np.array(
            [cam_model.f,cam_model.cx,cam_model.cy,cam_model.fov]
        )
        return intrinsics, None,cam_model.model
    elif cam_model.model =='PINHOLE':
        intrinsics = np.array(
            [cam_model.fx,cam_model.fy,cam_model.cx,cam_model.cy,cam_model.fovx,cam_model.fovy]
        )
        return intrinsics, None,cam_model.model
    elif cam_model.model == 'SIMPLE_RADIAL':
        intrinsics = np.array(
            [cam_model.f,cam_model.cx,cam_model.cy]
        )
        return intrinsics, {'k':cam_model.k,'k1':0.,'k2':0.,'p1':0.,'p2':0},cam_model.model
    else:
        raise NotImplementedError(f'camera model {cam_model.model} not implemented yet')
    
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    camera_model:NamedTuple
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def get_intrinsic_params(camera):
    if camera['model'] == 'SIMPLE_RADIAL':
        f = camera['intrinsics'][0]
        cx = camera['intrinsics'][1]
        cy = camera['intrinsics'][2]
        k = camera['intrinsics'][3]
        return CAM_SIMPLE_RADIAL(f,cx,cy,k,'SIMPLE_RADIAL')
    elif camera['model'] == 'RADIAL':
        f = camera['intrinsics'][0]
        cx = camera['intrinsics'][1]
        cy = camera['intrinsics'][2]
        k1 = camera['intrinsics'][3]
        k2 = camera['intrinsics'][3]
        return CAM_RADIAL(f,cx,cy,k1,k2,'RADIAL')
    elif camera['model'] == 'SIMPLE_PINHOLE':
        f = camera['intrinsics'][0]
        cx = camera['intrinsics'][1]
        cy = camera['intrinsics'][2]
        return CAM_SIMPLE_PINHOLE(f,cx,cy,'SIMPLE_PINHOLE')
    elif camera['model'] == 'PINHOLE':
        fx = camera['intrinsics'][0]
        fy = camera['intrinsics'][1]
        cx = camera['intrinsics'][2]
        cy = camera['intrinsics'][3]
        fovx = focal2fov(fx,camera['W'])
        fovy = focal2fov(fy,camera['H'])
        return CAM_PINHOLE(fx,fy,cx,cy,'PINHOLE',fovx,fovy) 
    elif camera['model'] == 'OPENCV':
        fx = camera['intrinsics'][0]
        fy = camera['intrinsics'][1]
        cx = camera['intrinsics'][2]
        cy = camera['intrinsics'][3]
        k1 = camera['intrinsics'][4]
        k2 = camera['intrinsics'][5]
        p1 = camera['intrinsics'][6]
        p2 = camera['intrinsics'][7]
        return CAM_OPENCV(fx,fy,cx,cy,k1,k2,p1,p2,'OPENCV')
    else:
        cam_type = camera['model']
        raise Exception(f'Camera type { cam_type } is not supported yet')

def get_RT(cam):
    c2w = torch.inverse(cam['w2c'])
    R, T = c2w[:3, :3], c2w[:3, 3:]
    R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1) # from RDF(colmap,OpenGL) to Left-Up-Forward (Pytorch3d) for Rotation
    
    new_c2w = torch.cat([R, T], 1)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]]).cuda()), 0))
    R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3]

    return R.cpu().numpy(),T.cpu().numpy()

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapCameras(path_to_colmap,images_folder):
    cam_infos_unsorted = []
    cameras, _ = read_cameras_from_sparse(path_to_colmap)
    # p3d = cameras['Points3d']
    # TODO: Support more camera 
    for cam in cameras:
        # c,h,w = setup_camera(cam,downsample_rate=downsample_rate) # Convert to Pytorch3d Camera
        camera_id = cam['camera_id']
        # Handle Intrinsic
        R,T = get_RT(cam)            
        img_path = os.path.join(path_to_colmap,images_folder,cam['img_name'])
        img = PIL_IMAGE.open(img_path)
        camera_model = get_intrinsic_params(cam)
        cam_info = CameraInfo(uid=cam['camera_id'],
                    R=R,T=T,
                    camera_model=camera_model,
                    image=img,
                    image_path=img_path,
                    image_name=cam['img_name'],
                    width=cam['W'],
                    height = cam['H'])
        cam_infos_unsorted.append(cam_info)
    return cam_infos_unsorted

def readColmapSceneInfo(path, images, eval, llffhold=8):
    # Similar Implementation but different coordinate system
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(path,reading_dir)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    xyz = []
    rgb = []
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            p3ds = read_points3D_binary(bin_path)
        except:
            p3ds= read_points3D_text(txt_path)
        for p in p3ds.values():
            xyz.append(p.xyz)
            rgb.append(p.rgb/255.0) 
        storePly(ply_path, np.array(xyz), np.array(rgb))
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)    
    return scene_info

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        color =  np.random.random((num_pts, 3)) 
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(color), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(color) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# OPENGL BLENDER RUB, PY3D LUF, COLMAP RDF
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, [0,2]] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = PIL_IMAGE.open(image_path)

            # im_data = np.array(image.convert("RGBA"))

            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = PIL_IMAGE.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            fy = fov2focal(fovy,image.size[1])
            fx = fov2focal(fovx,image.size[0])
            FovY = fovy 
            FovX = fovx
            cx = image.size[1]/2 # !! Assume at center
            cy = image.size[0]/2 
            camera_model = CAM_PINHOLE(fx,fy,cx,cy,'PINHOLE',FovY,FovX)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, camera_model=camera_model, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo, 
    "Blender" : readNerfSyntheticInfo,
}