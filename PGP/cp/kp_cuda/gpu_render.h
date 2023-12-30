#ifndef KP_NEW_CUDA_GPU_RENDER_CUH
#define KP_NEW_CUDA_GPU_RENDER_CUH

#include <string>
#include <fstream>
#include "model.h"
#include "TexturePlane.h"

using namespace std;

struct Light {
    float3 position;
    float3 color;
    float intensity;
};

struct Scene{
    int n_objects;
    int n_lights;
    int rays_per_pixel;
    int render_max_depth;
    int n_rays;
    int n_next_rays;
    int width;
    int height;
    GModel ** world;
    Light * lights;
    GTexturePlane * pl;
};

struct Camera{
    float3 eye;
    float3 gaze;
    int pixel_width;
    int pixel_height;
    int field_of_view;
    float aspectRatio;
    float scale;

    __host__ __device__ Camera() {printf("camera ctr\n");}
    __host__ __device__ Camera(float rc, float phic, float zc, float rn, float phin,
                               float zn, int px_w, int px_h, int fov);

    __host__ __device__ Matrix44 cameraToWorld() const;
};

__global__ void kernel_fill_rays_depth0(Ray * rays1, Camera camera, Scene * sc);
__global__ void kernel_cast(float3 *fb, Ray * rays1, Ray * rays2, Scene * sc);
__global__ void kernel_rays_copy(Ray * rays1, Ray * rays2, Scene * sc);
__global__ void  kernel_print_rays1(Ray * rays1, int n);
void save_frame(float3 *fb, int width, int height, string filename);

__global__ void gpu_ssaa(float3 *dst, float3 *src, Scene *sc);

__device__ float3 cast(Ray * rays1, Ray * rays2, int ray_idx, Scene *sc);
__device__ bool scene_intersect(Ray ray, Scene  *sc, HitPoint &best_hp);


#endif //KP_NEW_CUDA_GPU_RENDER_CUH
