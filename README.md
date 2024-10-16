# Triplet: Triangle Patchlet for Physics-Based Inverse Rendering and Scene Parameter Approximation

At present, this repository contains the code to evaluate the results of:
- [x] Rasterization - BlinnPhong
- [x] Rastierzation - Cook-Torrance

Currently, the code is written for proof-of-concept (POC) purposes in PyTorch. It may be slow and consume more memory than necessary. Some of the code will be rewritten using PyTorch extensions.

## What‘s next:

Mention in the paper, will be done:
- [ ] Efficiency and Quality Tuning: Different light sources and shaders exhibit varied behaviors during training due to their unique physical assumptions. As a result, tuning the training parameters for each light-shader combination is necessary. However, in most cases, the Cook-Torrance and Blinn-Phong models provide sufficiently high performance, and we focus on optimizing these now.
- [ ] A new rasterizer for Triplet that will be faster and require less VRAM (or exploration of other frameworks like Nvdiffrast).
- [ ] A CUDA-based shader framework that is faster and requires less VRAM (potentially implemented in PyCUDA for the convenience custom shaders).
- [ ] Ray tracing integration.
- [ ] CUDA implementation of all regularization terms (to save VRAM).
- [ ] Filtering material properties along ring neighbors.
- [ ] A mesh extraction algorithm designed for Triplet (temporarily, you can use the TSDF method modified from [1].see  [Here](scene/__init__.py#L209).
- [ ] Topology optimization/adaptive local densification on the mesh graph.
- [ ] Support for additional datasets.
- [ ] Interactive viewer.
- [ ] Material extraction.

Potential Future Work (Quality Improvements, Not Essential in Many Scenarios)
- [ ] Subsurface scattering (for those interested in more complex effects like those in Avatar, see [2][3]).
- [ ] Experiments with modern physically-based materials, especially anisotropic materials.

The code framework is forked from [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/README.md?plain=1)

## Results
Increasing the number of faces per pixel experimentally improves reconstruction quality. The current configuration is a low-end version. However, many optimizations (especially for memory usage and efficiency) are still pending. Full-power testing will be conducted once possible (i.e., with an RTXA6000 or the new rasterizer).

<details>
<summary><span style="font-weight: bold;"> Vertex-based SH lights </span></summary>
  
### NeRF synthetic dataset [4]
Faces_Per_Pixels: 20, grad_threhold=7.5e-5, sh_degree=5, beta (0.5,0.999)
| Method                     | Chair | Drums | Ficus | HotDog | Lego | Materials | Mic | Ship (1e-4) |
| ---------------------------| ------| ----- | ----- | ------ | ---- | --------- |---- |----- |
| Rasterization/BlinnPhong   |       |       |       |        |      |           |     |      |
| Rasterization/CookTorrance | 32.99 | 24.79 | 29.69 | 34.33  |29.81 |   27.12   |33.28|25.89 |
| RayTrace/BlinnPhong        |       |       |       |        |      |           |     |      |
| RayTrace/CookTorrance      |       |       |       |        |      |           |     |      |

Faces_Per_Pixels: 30, grad_threhold=7.5e-5, sh_degree=5, beta (0.5,0.999)
| Method                     | Chair | Drums | Ficus | HotDog | Lego | Materials | Mic | Ship |
| ---------------------------| ------| ----- | ----- | ------ | ---- | --------- |---- |----- |
| Rasterization/BlinnPhong   |       |       |       |        |      |           |     |      |
| Rasterization/CookTorrance |       |25.91  |       |        |      |           |     |      |
| RayTrace/BlinnPhong        |       |       |       |        |      |           |     |      |
| RayTrace/CookTorrance      |       |       |       |        |      |           |     |      |

Faces_Per_Pixels: 50, grad_threhold=7.5e-5, sh_degree=5, beta (0.5,0.999)
| Method                     | Chair | Drums | Ficus | HotDog | Lego | Materials | Mic | Ship |
| ---------------------------| ------| ----- | ----- | ------ | ---- | --------- |---- |----- |
| Rasterization/BlinnPhong   |       |       |       |        |      |           |     |      |
| Rasterization/CookTorrance |       |  |       |        |      |           |     |      |
| RayTrace/BlinnPhong        |       |       |       |        |      |           |     |      |
| RayTrace/CookTorrance      |       |       |       |        |      |           |     |      |

### Mip-NeRF360 dataset v2[5] 
Faces_Per_Pixels: 20, grad_threhold=1e-4 , sh_degree=3, random_background =True, no regulation terms, compensate_random_Point=True
| Method                     | Garden | Bicycle | Bonsai | Counter | Kitchen(sh_degree=1) | Room | Stump |
| ---------------------------| -------| ------- | ------ | ------- | ------- | ---- |------ |
| Rasterization/BlinnPhong   | 22.14  |20.93    |23.98   | 23.60   | 23.62  |23.97  |19.16   |
| Rasterization/CookTorrance |        |         |        |         |         |      |       |
| RayTrace/BlinnPhong        |        |         |        |         |         |      |       |
| RayTrace/CookTorrance      |        |         |        |         |         |      |       |

Faces_Per_Pixels: 40, grad_threhold=1e-4 , sh_degree=2, random_background =True, no regulation terms, compensate_random_Point=True
| Method                     | Garden | Bicycle | Bonsai | Counter | Kitchen | Room | Stump |
| ---------------------------| -------| ------- | ------ | ------- | ------- | ---- |------ |
| Rasterization/BlinnPhong   |        |         |        |         |         |      |       |
| Rasterization/CookTorrance |        |         |        |         |         |      |       |
| RayTrace/BlinnPhong        |        |         |        |         |         |      |       |
| RayTrace/CookTorrance      |        |         |        |         |         |      |       |

</details>
<br>

<details>
<summary><span style="font-weight: bold;">SH EnvMap</span></summary>
SH_degree = 9

### NeRF synthetic dataset 
Faces_Per_Pixels: 20, grad_threhold=7.5e-5 
| Method                     | Chair | Drums | Ficus | HotDog | Lego | Materials | Mic | Ship (1e-4) |
| ---------------------------| ------| ----- | ----- | ------ | ---- | --------- |---- |----- |
| Rasterization/BlinnPhong   |       |       |       |        |      |           |     |      |
| Rasterization/CookTorrance |  |  |  |   | |     || |
| RayTrace/BlinnPhong        |       |       |       |        |      |           |     |      |
| RayTrace/CookTorrance      |       |       |       |        |      |           |     |      |

Faces_Per_Pixels: 30 grad_threhold=7.5e-5 
| Method                     | Chair | Drums | Ficus | HotDog | Lego | Materials | Mic | Ship |
| ---------------------------| ------| ----- | ----- | ------ | ---- | --------- |---- |----- |
| Rasterization/BlinnPhong   |       |       |       |        |      |           |     |      |
| Rasterization/CookTorrance |       | |       |        |      |           |     |      |
| RayTrace/BlinnPhong        |       |       |       |        |      |           |     |      |
| RayTrace/CookTorrance      |       |       |       |        |      |           |     |      |

Faces_Per_Pixels: 50 grad_threhold=7.5e-5 
| Method                     | Chair | Drums | Ficus | HotDog | Lego | Materials | Mic | Ship |
| ---------------------------| ------| ----- | ----- | ------ | ---- | --------- |---- |----- |
| Rasterization/BlinnPhong   |       |       |       |        |      |           |     |      |
| Rasterization/CookTorrance |       |  |       |        |      |           |     |      |
| RayTrace/BlinnPhong        |       |       |       |        |      |           |     |      |
| RayTrace/CookTorrance      |       |       |       |        |      |           |     |      |

### Mip-NeRF360 dataset v2
Faces_Per_Pixels: 20, grad_threhold=2e-4 ,random_background =True
| Method                     | Garden | Bicycle | Bonsai | Counter | Kitchen | Room | Stump |
| ---------------------------| -------| ------- | ------ | ------- | ------- | ---- |------ |
| Rasterization/BlinnPhong   |        |         |        |         |         |      |       |
| Rasterization/CookTorrance |        |         |        |         |         |      |       |
| RayTrace/BlinnPhong        |        |         |        |         |         |      |       |
| RayTrace/CookTorrance      |        |         |        |         |         |      |       |

Faces_Per_Pixels: 40
| Method                     | Garden | Bicycle | Bonsai | Counter | Kitchen | Room | Stump |
| ---------------------------| -------| ------- | ------ | ------- | ------- | ---- |------ |
| Rasterization/BlinnPhong   |        |         |        |         |         |      |       |
| Rasterization/CookTorrance |        |         |        |         |         |      |       |
| RayTrace/BlinnPhong        |        |         |        |         |         |      |       |
| RayTrace/CookTorrance      |        |         |        |         |         |      |       |

</details>
<br>

<details>
<summary><span style="font-weight: bold;">Point Lights</span></summary>
Faces_Per_Pixels: 20, grad_threhold=7.5e-5 
### NeRF synthetic dataset 
  
| Method                     | Chair | Drums | Ficus | HotDog | Lego | Materials | Mic | Ship (1e-4) |
| ---------------------------| ------| ----- | ----- | ------ | ---- | --------- |---- |----- |
| Rasterization/BlinnPhong   |       |       |       |        |      |           |     |      |
| Rasterization/CookTorrance |  | |  |   | |      || |
| RayTrace/BlinnPhong        |       |       |       |        |      |           |     |      |
| RayTrace/CookTorrance      |       |       |       |        |      |           |     |      |

Faces_Per_Pixels: 30 grad_threhold=7.5e-5 
| Method                     | Chair | Drums | Ficus | HotDog | Lego | Materials | Mic | Ship |
| ---------------------------| ------| ----- | ----- | ------ | ---- | --------- |---- |----- |
| Rasterization/BlinnPhong   |       |       |       |        |      |           |     |      |
| Rasterization/CookTorrance |       | |       |        |      |           |     |      |
| RayTrace/BlinnPhong        |       |       |       |        |      |           |     |      |
| RayTrace/CookTorrance      |       |       |       |        |      |           |     |      |

Faces_Per_Pixels: 50 grad_threhold=7.5e-5 
| Method                     | Chair | Drums | Ficus | HotDog | Lego | Materials | Mic | Ship |
| ---------------------------| ------| ----- | ----- | ------ | ---- | --------- |---- |----- |
| Rasterization/BlinnPhong   |       |       |       |        |      |           |     |      |
| Rasterization/CookTorrance |       |  |       |        |      |           |     |      |
| RayTrace/BlinnPhong        |       |       |       |        |      |           |     |      |
| RayTrace/CookTorrance      |       |       |       |        |      |           |     |      |

### Mip-NeRF360 dataset v2
Faces_Per_Pixels: 20, grad_threhold=2e-4 , random_background =True
| Method                     | Garden | Bicycle | Bonsai | Counter | Kitchen | Room | Stump |
| ---------------------------| -------| ------- | ------ | ------- | ------- | ---- |------ |
| Rasterization/BlinnPhong   |        |         |        |         |         |      |       |
| Rasterization/CookTorrance |        |         |        |         |         |      |       |
| RayTrace/BlinnPhong        |        |         |        |         |         |      |       |
| RayTrace/CookTorrance      |        |         |        |         |         |      |       |

Faces_Per_Pixels: 40
| Method                     | Garden | Bicycle | Bonsai | Counter | Kitchen | Room | Stump |
| ---------------------------| -------| ------- | ------ | ------- | ------- | ---- |------ |
| Rasterization/BlinnPhong   |        |         |        |         |         |      |       |
| Rasterization/CookTorrance |        |         |        |         |         |      |       |
| RayTrace/BlinnPhong        |        |         |        |         |         |      |       |
| RayTrace/CookTorrance      |        |         |        |         |         |      |       |

</details>
<br>


<details>
<summary><span style="font-weight: bold;">Direction Lights</span></summary>
Faces_Per_Pixels: 20, grad_threhold=7.5e-5 
### NeRF synthetic dataset 
| Method                     | Chair | Drums | Ficus | HotDog | Lego | Materials | Mic | Ship (1e-4) |
| ---------------------------| ------| ----- | ----- | ------ | ---- | --------- |---- |----- |
| Rasterization/BlinnPhong   |       |       |       |        |      |           |     |      |
| Rasterization/CookTorrance |  | |  |   | |     | | |
| RayTrace/BlinnPhong        |       |       |       |        |      |           |     |      |
| RayTrace/CookTorrance      |       |       |       |        |      |           |     |      |

Faces_Per_Pixels: 30 grad_threhold=7.5e-5 
| Method                     | Chair | Drums | Ficus | HotDog | Lego | Materials | Mic | Ship |
| ---------------------------| ------| ----- | ----- | ------ | ---- | --------- |---- |----- |
| Rasterization/BlinnPhong   |       |       |       |        |      |           |     |      |
| Rasterization/CookTorrance |       |  |       |        |      |           |     |      |
| RayTrace/BlinnPhong        |       |       |       |        |      |           |     |      |
| RayTrace/CookTorrance      |       |       |       |        |      |           |     |      |

Faces_Per_Pixels: 50 grad_threhold=7.5e-5 
| Method                     | Chair | Drums | Ficus | HotDog | Lego | Materials | Mic | Ship |
| ---------------------------| ------| ----- | ----- | ------ | ---- | --------- |---- |----- |
| Rasterization/BlinnPhong   |       |       |       |        |      |           |     |      |
| Rasterization/CookTorrance |       |  |       |        |      |           |     |      |
| RayTrace/BlinnPhong        |       |       |       |        |      |           |     |      |
| RayTrace/CookTorrance      |       |       |       |        |      |           |     |      |

### Mip-NeRF360 dataset v2
Faces_Per_Pixels: 20, grad_threhold=2e-4 , random_background =True
| Method                     | Garden | Bicycle | Bonsai | Counter | Kitchen | Room | Stump |
| ---------------------------| -------| ------- | ------ | ------- | ------- | ---- |------ |
| Rasterization/BlinnPhong   |        |         |        |         |         |      |       |
| Rasterization/CookTorrance |        |         |        |         |         |      |       |
| RayTrace/BlinnPhong        |        |         |        |         |         |      |       |
| RayTrace/CookTorrance      |        |         |        |         |         |      |       |

Faces_Per_Pixels: 40
| Method                     | Garden | Bicycle | Bonsai | Counter | Kitchen | Room | Stump |
| ---------------------------| -------| ------- | ------ | ------- | ------- | ---- |------ |
| Rasterization/BlinnPhong   |        |         |        |         |         |      |       |
| Rasterization/CookTorrance |        |         |        |         |         |      |       |
| RayTrace/BlinnPhong        |        |         |        |         |         |      |       |
| RayTrace/CookTorrance      |        |         |        |         |         |      |       |

</details>
<br>

## Installation:
### Clone the project
```shell
git clone git@github.com:RANDO11199/ParticleFieldDuality.git --recursive
```
### Install Pytorch3d in the submodules
```shell
git clone git@github.com:RANDO11199/Pytorch3d4triplet.git
cd pytorch3d4triplet
pip install -e .
```
For more details on installing PyTorch3D, see the official [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#building--installing-from-source)
### Install other dependencies:
```shell
pip install -r requirements.txt
```

## Usage
### Training
```shell
python train.py -s <path to your colmap or Synthetic NeRF dataset > 
```
## Extra
Currently, I am looking for research opportunities (Job/PhD). It would be a great help if you could ⭐ Star my project. For any questions, opportunities, or cooperation, please contact me at jiajie.y@wustl.edu.

If you find the code or paper helpful, please consider citing me!:D

## Reference
[1] https://github.com/hbb1/2d-gaussian-splatting

[2] https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-14-advanced-techniques-realistic-real-time-skin

[3] Borshukov, G., and J. P. Lewis. "Realistic human face rendering for." The Matrix Reloaded”,” in ACM SIGGRAPH 2003 Conference Abstracts and Applications (Sketch). 2003.

[4] Mildenhall, Ben, et al. "Nerf: Representing scenes as neural radiance fields for view synthesis." Communications of the ACM 65.1 (2021): 99-106.

[5] Barron, Jonathan T., et al. "Mip-nerf 360: Unbounded anti-aliased neural radiance fields." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[6] Ravi, Nikhila, et al. "Accelerating 3d deep learning with pytorch3d." arXiv preprint arXiv:2007.08501 (2020).
