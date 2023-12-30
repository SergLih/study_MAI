#include <limits>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <pthread.h>
#include <set>
#include "vector.hpp"
#include "model.h"
#include "sphere.h"
#include "Matrix.hpp"
#include "TexturePlane.h"
#include "ssaa.h"

using namespace std;

struct Params {
    int i_px;
    Ray ray;
};

vector<Vec3f> framebuffer;
vector<IHittable*> objects;
vector<Light>  lights;
pthread_t * threads;

void* cpuRayCast(void *dummyPtr) {
    Params *p = (Params *) dummyPtr;
    //cout << "T #" << p->threadIdx << ": ";
    if(p->i_px == -1) {
    } else {
        framebuffer[p->i_px] = p->ray.cast(objects, lights);
    }
    return nullptr;
}

void render(const Camera &camera, const vector<IHittable*> &objects, string filename) {
    const int width    = camera.pixel_width * Ray::rays_per_pixel;
    const int height   = camera.pixel_height * Ray::rays_per_pixel;
    const int fov      = camera.field_of_view;
    framebuffer.resize(width*height);
    float imageAspectRatio = width/(float)height;

    Matrix44f cameraToWorld = camera.cameraToWorld();
    //cout << cameraToWorld;
    float scale = tan((fov * 0.5));
    Vec3f orig, dir;
    cameraToWorld.multVecMatrix(Vec3f(), orig);

    int n_threads_cpu = 128 * 64;
    threads = new pthread_t[n_threads_cpu];
    Params * params = new Params[n_threads_cpu];

    int i_px = 0;
    while(i_px < width * height) {
        for (int i_thr = 0; i_thr < n_threads_cpu; ++i_thr) {
            if(i_px+i_thr >= width*height)
                params[i_thr].i_px = -1;
            else {
                int j = (i_px+i_thr) / width, i = (i_px+i_thr) % width;
                float x = (2 * (i + 0.5f) / (float) width - 1) * imageAspectRatio * scale;
                float y = (1 - 2 * (j + 0.5f) / (float) height) * scale;
                cameraToWorld.multDirMatrix(Vec3f(x, y, -1), dir);
                dir = dir.normalize();
                params[i_thr].i_px = (i_px+i_thr);
                params[i_thr].ray = Ray(orig, dir);;
                //cout <<  params[i_thr].ray.orig << "\t" << params[i_thr].ray.dir << "\n";
            }
            pthread_create(&threads[i_thr], NULL, cpuRayCast, (void *) &params[i_thr]);
        }

        for (int i_thr = 0; i_thr < n_threads_cpu; ++i_thr) {
            pthread_join(threads[i_thr], NULL);
        }
        i_px += n_threads_cpu;
    }

    delete [] threads;
    delete[] params;

    cpu_ssaa(framebuffer, width, height, Ray::rays_per_pixel, 128);

    ofstream ofs; // save the framebuffer to file
    ofs.open(filename);
    ofs << "P6\n" << camera.pixel_width << " " << camera.pixel_height << "\n255\n";
    for (size_t i = 0; i < camera.pixel_height*camera.pixel_width; ++i) {
        Vec3f &c = framebuffer[i];
        float max = std::max(c[0], std::max(c[1], c[2]));
        if (max>1) c = c*(1./max);
        // cout << framebuffer[i][0] << " "<< framebuffer[i][1] << " "<< framebuffer[i][2] << "\t"
        //         << (int)(255 * std::max(0.f, std::min(1.f, framebuffer[i][0]))) << "\t"
        //         << (int)(255 * std::max(0.f, std::min(1.f, framebuffer[i][1]))) << "\t"
        //         << (int)(255 * std::max(0.f, std::min(1.f, framebuffer[i][2]))) << "\n";
        for (size_t j = 0; j<3; j++) {
            ofs << (char)(255 * std::max(0.f, std::min(1.f, framebuffer[i][j])));
        }
    }
    ofs.close();
}

void update_objects_and_lights(Model*model)
{
    objects.push_back(model);
    objects.insert(objects.end(), model->lights_spheres.begin(), model->lights_spheres.end());
}


int main() {
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
    Vec3f center1, center2, center3, color1, color2, color3, tex_color, light_coord, light_color;
    float radius1, radius2, radius3, coef_ref1, coef_ref2, coef_ref3, coef_trans1, coef_trans2, coef_trans3, tex_coef_ref;
    int n_lights1, n_lights2, n_lights3, n_lights_scene;
    Vec3f p[4];
    string texture_path;

    cin >> center1 >> color1 >> radius1 >> coef_ref1 >> coef_trans1 >> n_lights1;
    cin >> center2 >> color2 >> radius2 >> coef_ref2 >> coef_trans2 >> n_lights2;
    cin >> center3 >> color3 >> radius3 >> coef_ref3 >> coef_trans3 >> n_lights3;
    cin >> p[0] >> p[1] >> p[2] >> p[3] >> texture_path >> tex_color >> tex_coef_ref;

    for (int j = 0; j < 4; ++j) {
        p[j] = !p[j];
    }

//    Количество (не более четырех) и параметры источников света: положение и цвет.
    cin >> n_lights_scene;
    for (int k = 0; k < n_lights_scene; ++k) {
        cin >> light_coord >> light_color;
        lights.push_back(Light(light_coord, light_color, 10));
    }


    int tmp;
    cin >> Ray::max_depth >> tmp;
    Ray::rays_per_pixel = (int)sqrt(tmp);

    TexturePlane * pl = new TexturePlane(p, texture_path, 0.05, tex_color, tex_coef_ref);
    objects.push_back(pl);


    Material mat1(coef_trans1, coef_ref1, color1);
    Material mat2(coef_trans2, coef_ref2, color2);
    Material mat3(coef_trans3, coef_ref3, color3);

    Material      glass(1.5, Vec4f(0.0,  0.5, 0.1, 0.8), Vec3f(0.6, 0.7, 0.8),  125.);
    Material red_rubber(1.0, Vec4f(0.9,  0.1, 0.0, 0.0), Vec3f(0.3, 0.1, 0.1),   10.);
    Material     mirror(1.0, Vec4f(0.0, 10.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 1425.);
    Material      ivory(1.0, Vec4f(0.9,  0.1, 0.0, 0.0), Vec3f(0.4, 0.4, 0.3),   5.);


    Model* oct = new Model("plato/oct1.obj", mat1, n_lights1, !center1, radius1);
    Model* cube = new Model("plato/cube1.obj", mat2, n_lights2, !center2, radius2);
    Model* tetr = new Model("plato/tetr1.obj", mat3, n_lights3, !center3, radius3);
    update_objects_and_lights(oct);
    update_objects_and_lights(cube);
    update_objects_and_lights(tetr);
    objects.push_back(new Sphere(Vec3f(0,0,0), Model::spot_light_radius, Model::spot_light_material));
    lights.push_back(Light(Vec3f(0,0,0), Vec3f(1, 1, 1), Model::spot_light_intensity));

    Model * floor = new Model("plato/floor.obj", ivory,3);
    objects.push_back(floor);

    float h = 2 * M_PI / n_frames;
    float t = 0, rc, phic, zc, rn, phin, zn;
    for (int i = 0; i < n_frames; ++i) {
        rc   = rc0 + Arc*sin(wphic*t + prc);
        zc   = z0c + Azc*sin(wphic * t + pzc);
        phic = phi0c + wphic * t;
        rn   = rn0 + Arn*sin(wphin*t + prn);
        zn   = z0n + Azn*sin(wphin * t + pzn);
        phin = phi0n + wphin * t;
        Camera camera2(rc, phic, zc, rn, phin, zn, pixel_width, pixel_height, field_of_view);
        auto pos = path.find("%d");
        string filename = path.substr(0, pos) + to_string(i) + path.substr(pos+2);
        //cout << filename << endl;
        time_t start0 = clock();
        render(camera2, objects, filename);
        time_t end0 = clock();
        fprintf(stderr, "CPU: ready\n");
        fprintf(stderr, "Working time:     %f sec.\n", (double)(end0 - start0) / (double)CLOCKS_PER_SEC);
        t += h;
        //break;
    }

    for(IHittable *& obj: objects)
        delete obj;

    return 0;
}
