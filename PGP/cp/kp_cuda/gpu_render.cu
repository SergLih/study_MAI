#include "gpu_render.h"

using namespace std;


void save_frame(float3 *buf, int pixel_width, int pixel_height, string filename) {
    ofstream ofs; // save the framebuffer to file
    ofs.open(filename);
    ofs << "P6\n" << pixel_width << " " << pixel_height << "\n255\n";
    for (size_t i = 0; i < pixel_height*pixel_width; ++i) {
        ofs << to_bytes(buf[i]);
    }
    ofs.close();
}

__device__ void print(const float3 &v) {
    //printf("%f %f %f\t", v.x, v.y, v.z);
}

__global__ void kernel_fill_rays_depth0(Ray * rays1, Camera camera, Scene * sc) {
    ////printf("kernel_fill_rays_depth0 started\n");
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    float3 orig = make_float3(0, 0, 0);
    float3 dir = make_float3(0, 0, 0);

    Matrix44 cam2world = camera.cameraToWorld();
    float aspectRatio = camera.aspectRatio;
    float scale = camera.scale;
    cam2world.multVecMatrix(dir, orig);

    for(int j=idy; j < sc->height; j+=offsety) {
        for(int i=idx; i < sc->width; i+=offsetx) {
            float x = (2 * (i + 0.5f) / (float) sc->width - 1) * aspectRatio * scale;
            float y = (1 - 2 * (j + 0.5f) / (float) sc->height) * scale;
            cam2world.multDirMatrix(make_float3(x, y, -1), dir);
            dir = normalize(dir);
            rays1[j*sc->width + i] = Ray(orig, dir, 1, i, j);
        }
    }
}

__global__ void  kernel_print_rays1(Ray * rays1, int n) {
    for(int t = 0; t < n; t++) {
        Ray ray  = rays1[t];
        printf("=%d\t%f %f %f -> %f %f %f\t| %d %d\td=%d\n",
               t, ray.orig.x, ray.orig.y, ray.orig.z, ray.dir.x, ray.dir.y, ray.dir.z, ray.i, ray.j,
               ray.depth);
    }
}

__global__ void kernel_cast(float3 *fb, Ray * rays1, Ray * rays2, Scene * sc) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    ////printf("cast %d | %d \n", idx, sc->n_rays);

    for(int t = idx; t < sc->n_rays; t+=offsetx) {
        fb[rays1[t].j*sc->width + rays1[t].i] =
                fb[rays1[t].j*sc->width + rays1[t].i] + cast(rays1, rays2, t, sc) * rays1[t].coef;
    }
}

__global__ void kernel_rays_copy(Ray * rays1, Ray * rays2, Scene * sc) {
    ////printf(">> copy: n1 = %d, n2 = %d | %016llx %016llx\n", sc->n_rays, 2*sc->n_rays, rays1, rays2);
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    int k2 = 2 * sc->n_rays;
    for (int t = idx; t < k2; t+=offsetx) {
        if (rays2[t].depth > 0) {
            rays1[atomicAdd(&sc->n_next_rays, 1)] = rays2[t];
        }
    }
}

__global__ void gpu_ssaa(float3 *dst, float3 *src, Scene *sc) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int c = sc->rays_per_pixel;

    int w_in = sc->width;
    int h_in = sc->height;
    int w_out = w_in/c;
    int h_out = h_in/c;

    for (int j_out = idy; j_out < h_out; j_out += offsety) {
        for (int i_out = idx; i_out < w_out; i_out += offsetx) {
            int j_in = j_out * c, i_in = i_out * c, n = 0;
            float3 s = make_float3(0., 0., 0.);
            for(int j = 0; j < c; j++) {
                for(int i = 0; i < c; i++) {

                    if(j_in + j >= h_in || i_in + i >= w_in)
                        continue;
                    n++;
                    s = s + src[i_in + i + (j_in + j) * w_in];
                }
            }
            s = s * (1.0 / n);
            dst[j_out * w_out + i_out] = s;
        }
    }
}

__device__ float3 cast(Ray * rays1, Ray * rays2, int ray_idx, Scene *sc) {
    Ray ray = rays1[ray_idx];

//    //printf("%f %f %f -> %f %f %f | %d %d\t%d / %d\n",
//            ray.orig.x, ray.orig.y, ray.orig.z, ray.dir.x, ray.dir.y, ray.dir.z, ray.i, ray.j,
//            ray.depth, sc->render_max_depth);

    HitPoint hp, tmp_hp;
    bool res = scene_intersect(ray, sc, hp);
    if (ray.depth > sc->render_max_depth || !res) {
        return make_float3(0.0, 0.0, 0.0); // background color
    }
    else if(hp.material.diffuse_color == make_float3(1, 1, 1) ) {
        return make_float3(1, 1, 1);
    }

    float3 diffuse_light_intensity = make_float3(0,0,0);
    float3 specular_light_intensity = make_float3(0,0,0);

    for (size_t i=0; i<sc->n_lights; i++) {
        float3 light_dir = normalize(sc->lights[i].position - hp.point);
        float light_distance = norm(sc->lights[i].position - hp.point);
        float3 shadow_orig = light_dir * hp.N < 0 ? hp.point - hp.N * 1e-3 : hp.point + hp.N * 1e-3; // checking if the point lies in the shadow of the lights_coords[i]
        Ray tmp_ray(shadow_orig, light_dir, ray.i, ray.j, ray.depth+1);
        if (scene_intersect(tmp_ray, sc, tmp_hp)
            && tmp_hp.material.diffuse_color != make_float3(1, 1, 1)
            && norm((tmp_hp.point - tmp_ray.orig)) < light_distance)
            continue;

        diffuse_light_intensity = diffuse_light_intensity +
                                  (hp.material.diffuse_color % sc->lights[i].color)
                                  * sc->lights[i].intensity * dmax(0.f, light_dir * hp.N);

        specular_light_intensity = specular_light_intensity +
                                   (make_float3(1., 1., 1.) % sc->lights[i].color) *
                                   powf(dmax(0.f, -reflect(-light_dir, hp.N) * ray.dir),
                                        hp.material.specular_exponent) * sc->lights[i].intensity;
    }
    float3 result_color = diffuse_light_intensity * hp.material.albedo.x
                          + specular_light_intensity * hp.material.albedo.y;

    if(ray.depth < sc->render_max_depth) {
        atomicAdd(&sc->n_next_rays, 2);
        float3 reflect_dir = normalize(reflect(ray.dir, hp.N));
        float3 refract_dir = normalize(refract(ray.dir, hp.N, hp.material.refractive_index));
        float3 reflect_orig = reflect_dir * hp.N < 0 ? hp.point - hp.N * 1e-3 : hp.point + hp.N *  1e-3; // offset the original point to avoid occlusion by the object itself
        float3 refract_orig = refract_dir * hp.N < 0 ? hp.point - hp.N * 1e-3 : hp.point + hp.N * 1e-3;
////printf("[%016llx / %016llx - %016llx] b\t", &rays2[2 * ray_idx], rays2, rays2 + n_rays*2);
        if(ray.coef * hp.material.albedo.w > 0.01)
            rays2[2 * ray_idx] = Ray(refract_orig, refract_dir, ray.coef * hp.material.albedo.w,
                                    ray.i, ray.j, ray.depth + 1);
        if(ray.coef * hp.material.albedo.z > 0.01)
            rays2[2 * ray_idx + 1] = Ray(reflect_orig, reflect_dir, ray.coef * hp.material.albedo.z,
                                    ray.i, ray.j, ray.depth + 1);
            //printf("a\n");

    }
    return result_color;
}

__device__ bool scene_intersect(Ray ray, Scene  *sc, HitPoint &best_hp) {
    bool found = false, hit_plus;
    HitPoint cur_hp;
    best_hp.dist = 1e38;


    for (int i = 0; i < sc->n_objects; ++i) {
        hit_plus = sc->world[i]->hit(ray, cur_hp);
        if(hit_plus && cur_hp.dist < best_hp.dist) {
            best_hp = cur_hp; found = true;
        }
        for (int j = 0; j < sc->world[i]->n_lights; ++j) {
            hit_plus = sc->world[i]->light_spheres[j].hit(ray, cur_hp);
            if(hit_plus && cur_hp.dist < best_hp.dist) {
                best_hp = cur_hp; found = true;
            }
        }
    }

    hit_plus = sc->pl->hit(ray, cur_hp);
    if(hit_plus && cur_hp.dist < best_hp.dist) {
        best_hp = cur_hp; found = true;
        ////printf(" FLOOR: %f %f %f\n", best_hp.material.diffuse_color.x,
        //       best_hp.material.diffuse_color.y, best_hp.material.diffuse_color.z);
    }

    return found;
}


__device__ Matrix44 Camera::cameraToWorld() const {
    float3 u, v, w, t;
    t = make_float3(0,1,0);
    w = normalize((eye - gaze));
    u = normalize(cross(t, w));
    v = cross(w, u);
    float m[4][4] = {{u.x, u.y, u.z, 0},
                     {v.x, v.y, v.z, 0},
                     {w.x, w.y, w.z, 0},
                     {eye.x, eye.y, eye.z, 1}};
    return Matrix44(m);
}

Camera::Camera(float rc, float phic, float zc, float rn, float phin, float zn, int px_w, int px_h, int fov) {
    eye = make_float3(rc * cos(phic), zc, -rc * sin(phic));
    gaze = make_float3(rn * cos(phin), zn, -rn * sin(phin));
    pixel_width = px_w;
    pixel_height = px_h;
    field_of_view = fov * M_PI / 180;
    scale = tan((field_of_view * 0.5));
    aspectRatio = pixel_width/(float)pixel_height;
}
