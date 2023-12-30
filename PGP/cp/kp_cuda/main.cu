#include <limits>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <pthread.h>
#include <set>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include  <stdio.h>

#include "model.h"
#include "Matrix.h"
#include "TexturePlane.h"
#include "gpu_render.h"

#define CSC(call) do { \
	cudaError_t pixels = call;	\
	if (pixels != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(pixels)); \
		exit(0); \
	} \
} while (0)

using namespace std;

#pragma hd_warning_disable

void myCudaRealloc(Ray * &p, size_t old_size, size_t new_size) {
    //cout << "realloc\n";
    if(new_size > old_size) {
        CSC(cudaFree(p));
        CSC(cudaMalloc(&p, new_size));
    }
    CSC(cudaMemset(p, 0, new_size));
}


int main() {
    Scene *sc;
    cudaMallocManaged(&sc, sizeof(Scene));
    int n_frames, pixel_width, pixel_height, field_of_view;
    float rc0, Arc, wrc, prc, z0c, Azc, wzc, phi0c, wphic, pzc,
            rn0, Arn, wrn, prn, z0n, Azn, wzn, phi0n, wphin, pzn;
    string path;

    cin >> n_frames >> path;
    cin >> pixel_width >> pixel_height >> field_of_view;
    cin >> rc0 >> z0c >> phi0c >> Arc >> Azc >> wrc >> wzc >> wphic >> prc >> pzc
        >> rn0 >> z0n >> phi0n >> Arn >> Azn >> wrn >> wzn >> wphin >> prn >> pzn;

    //Параметры тел: центр тела, цвет (нормированный), радиус (подразумевается
    //радиус сферы в которую можно было бы вписать тело), коэффициент
    //отражения, коэффициент прозрачности, количество точечных источников света
    //на ребре
    float3 center1, center2, center3, color1, color2, color3, tex_color;
    float radius1, radius2, radius3, coef_ref1, coef_ref2, coef_ref3, coef_trans1, coef_trans2, coef_trans3, tex_coef_ref;
    int n_lights1, n_lights2, n_lights3;
    float3 *p = new float3[4];
    string texture_path;

    cin >> center1 >> color1 >> radius1 >> coef_ref1 >> coef_trans1 >> n_lights1;
    cin >> center2 >> color2 >> radius2 >> coef_ref2 >> coef_trans2 >> n_lights2;
    cin >> center3 >> color3 >> radius3 >> coef_ref3 >> coef_trans3 >> n_lights3;
    cin >> p[0] >> p[1] >> p[2] >> p[3] >> texture_path >> tex_color >> tex_coef_ref;

    for (int j = 0; j < 4; ++j) {
        p[j] = !p[j];
    }

//    Количество (не более четырех) и параметры источников света: положение и цвет.
    cin >> sc->n_lights;
    cudaMallocManaged(&sc->lights, sc->n_lights * sizeof(Light));
    for (int k = 0; k <sc->n_lights; ++k) {
        cin >> sc->lights[k].position >> sc->lights[k].color;
        sc->lights[k].intensity = 10;
        //printf("--- LIGHT: %d\t%f\n", k, sc->lights[k].intensity);
    }


    double tmp;
    cin >> sc->render_max_depth >> tmp;
    sc->rays_per_pixel = (int)sqrt(tmp);
    //cout << "ssaa coeff: " << sc->rays_per_pixel << "\n";

    sc->pl = createGTexturePlane(p, texture_path, 0.05, tex_color, tex_coef_ref);
    delete[] p;

    Material mat1(coef_trans1, coef_ref1, color1);
    Material mat2(coef_trans2, coef_ref2, color2);
    Material mat3(coef_trans3, coef_ref3, color3);

    Material      glass(1.5, make_float4(0.0,  0.5, 0.1, 0.8), make_float3(0.6, 0.7, 0.8),  125.);
    Material red_rubber(1.0, make_float4(0.9,  0.1, 0.0, 0.0), make_float3(0.3, 0.1, 0.1),   10.);
    Material     mirror(1.0, make_float4(0.0, 10.0, 0.8, 0.0), make_float3(1.0, 1.0, 1.0), 1425.);
    Material      ivory(1.0, make_float4(0.9,  0.1, 0.0, 0.0), make_float3(0.4, 0.4, 0.3),   5.);


    Model* oct = new Model("/home/sergey/MAI/PGP/kp_tracer/plato/oct1.obj", mat1, n_lights1, !center1, radius1);
    Model* cube = new Model("/home/sergey/MAI/PGP/kp_tracer/plato/cube1.obj", mat2, n_lights2, !center2, radius2);
    Model* tetr = new Model("/home/sergey/MAI/PGP/kp_tracer/plato/tetr1.obj", mat3, n_lights3, !center3, radius3);

    sc->n_objects = 3;
    cudaMallocManaged(&sc->world, sc->n_objects*sizeof(GModel*));
    sc->world[0] = oct->getGModel();
    sc->world[1] =cube->getGModel();
    sc->world[2] =tetr->getGModel();

    sc->width    = pixel_width * sc->rays_per_pixel;
    sc->height   = pixel_height * sc->rays_per_pixel;

    Ray *rays1, *rays2;
    float3 *buf_cpu = new float3[pixel_width * pixel_height];
    float3 *buf_gpu, *buf_gpu_ssaa;
    CSC(cudaMalloc(&buf_gpu, sizeof(float3) * sc->width * sc->height));
    CSC(cudaMalloc(&buf_gpu_ssaa, sizeof(float3) * pixel_width * pixel_height));

    float h = 2 * M_PI / n_frames;
    float t = 0, rc, phic, zc, rn, phin, zn;
    for (int i = 0; i < n_frames; ++i) {
        rc   = rc0 + Arc*sin(wphic*t + prc);
        zc   = z0c + Azc*sin(wphic * t + pzc);
        phic = phi0c + wphic * t;
        rn   = rn0 + Arn*sin(wphin*t + prn);
        zn   = z0n + Azn*sin(wphin * t + pzn);
        phin = phi0n + wphin * t;
        Camera camera = Camera(rc, phic, zc, rn, phin, zn, pixel_width, pixel_height, field_of_view);
        auto pos = path.find("%d");
        string filename = path.substr(0, pos) + to_string(i) + path.substr(pos+2);
        //cout << filename << endl;

        sc->n_rays = sc->width * sc->height;
        sc->n_next_rays = 0;

        CSC(cudaMalloc(&rays1, sizeof(Ray) * sc->n_rays));
        CSC(cudaMalloc(&rays2, sizeof(Ray) * sc->n_rays * 2));
        cudaEvent_t start, end;
        CSC(cudaEventCreate(&start));
        CSC(cudaEventCreate(&end));
        CSC(cudaEventRecord(start));
        kernel_fill_rays_depth0<<<32, dim3(32, 32)>>>(rays1, camera, sc);
        cudaGetLastError();
        int total_rays_frame = 0;
        while(true){
            //cout << "===================================";
            //kernel_print_rays1<<<1, 1>>>(rays1, sc->n_rays);
            kernel_cast<<<32, 512>>>(buf_gpu, rays1, rays2, sc);
            CSC(cudaDeviceSynchronize());
            //printf(">>after cast:  rays1: %d rays2: %d |  next=%d\n",
            //       sc->n_rays, sc->n_rays*2, sc->n_next_rays);

            total_rays_frame += sc->n_rays;

            if(sc->n_next_rays == 0)
                break;

            myCudaRealloc(rays1, sizeof(Ray) * sc->n_rays, sizeof(Ray) * sc->n_next_rays);
            //printf(">>before copy:  rays1: %d rays2: %d\n", sc->n_next_rays, sc->n_rays*2);
            sc->n_next_rays = 0;
            kernel_rays_copy<<<32, 256>>>(rays1, rays2, sc);
            cudaGetLastError();
            CSC(cudaDeviceSynchronize());
            myCudaRealloc(rays2, sizeof(Ray)*sc->n_rays*2, sizeof(Ray) * sc->n_next_rays*2);

            //printf(">>before next_iter:  rays1: %d rays2: %d\n", sc->n_next_rays, sc->n_next_rays*2);
            sc->n_rays = sc->n_next_rays;
            sc->n_next_rays = 0;
            //kernel_print_rays1<<<1, 1>>>(rays1, sc->n_rays);
            //break;
        }

        //SSAA
        gpu_ssaa<<<dim3(8, 8), dim3(32, 32)>>>(buf_gpu_ssaa, buf_gpu, sc);

        CSC(cudaEventRecord(end));
        CSC(cudaEventSynchronize(end));
        float time;
        CSC(cudaEventElapsedTime(&time, start, end));
        CSC(cudaEventDestroy(start));
        CSC(cudaEventDestroy(end));
        
        cout << "Number frame: " << i << " | time(ms): " << time << " | total number rays in frame: " << total_rays_frame << "\n\n";
        
CSC(cudaMemcpy(buf_cpu, buf_gpu_ssaa, sizeof(float3) * pixel_width * pixel_height, cudaMemcpyDeviceToHost));
        CSC(cudaMemset(buf_gpu, 0, sizeof(float3) * sc->width * sc->height));

        save_frame(buf_cpu, pixel_width, pixel_height, filename);

        CSC(cudaFree(rays1));
        CSC(cudaFree(rays2));
        t += h;
        //break;
    }

    delete [] buf_cpu;

    CSC(cudaFree(buf_gpu));
    CSC(cudaFree(buf_gpu_ssaa));
    CSC(cudaFree(sc));


    return 0;
}
