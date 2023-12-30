#ifndef KP_TRACER_H
#define KP_TRACER_H

#include "vector.hpp"
#include "Matrix.hpp"
class TexturePlane;
class Sphere;

struct Ray;

struct Material {
    Material(float r, const Vec4f &a, const Vec3f &color, float spec) : refractive_index(r), albedo(a), diffuse_color(color), specular_exponent(spec) {}
    Material() : refractive_index(1), albedo(1,0,0,0), diffuse_color(), specular_exponent() {}
    Material(float coef_refr, float coef_refl, const Vec3f &color):  diffuse_color(color)
    {
        refractive_index = coef_refr;
        if(coef_refl > 0.5){
            albedo = Vec4f(0.0,  0.5, 0.1, 0.8);
            specular_exponent = coef_refl*2000;
        } else {
            albedo = Vec4f(0.0, 10.0, 0.8, 0.0);
            specular_exponent = coef_refl*200;
        }
    }
    float refractive_index;
    Vec4f albedo;
    Vec3f diffuse_color;
    float specular_exponent;
};

struct Light {
    Light(Vec3f p, float i) {
        position = p;
        intensity = i;        
        color=Vec3f(1,1,1);
    }
    Light(Vec3f p, Vec3f c, float i) {
        position = p;
        intensity = i;
        color = c;  
    }
    Vec3f position;
    Vec3f color;
    float intensity;
};

class IHittable{
public:
    virtual bool hit(const Ray &ray, float &t, Vec3f &normal, Material &material) const = 0;
    virtual ~IHittable() = 0;
};

Vec3f reflect(const Vec3f &I, const Vec3f &N);
Vec3f refract(const Vec3f &I, const Vec3f &N, const float &refractive_index);

struct Ray{
    Vec3f orig;
    Vec3f dir;
    static int max_depth;
    static int rays_per_pixel;

    Ray();
    Ray(Vec3f orig, Vec3f dir): orig(orig), dir(dir) {}

    Vec3f get_point(float t) const {
        return orig + dir * t;
    }

    Vec3f cast(const std::vector<IHittable*> &objects, const std::vector<Light> &lights, size_t depth=0);
    bool scene_intersect(const std::vector<IHittable*> &objects, IHittable* &hit_obj, Vec3f &hitpoint, Vec3f &normal, Material &material);

};

std::ostream& operator<<(std::ostream& out, const Ray & r);

struct Edge{
    Vec3f start;
    Vec3f end;
};

struct Camera{
    Vec3f eye;
    Vec3f gaze;
    int pixel_width;
    int pixel_height;
    int field_of_view;

    Camera(float rc, float phic, float zc, float rn, float phin, float zn, int px_w, int px_h, int fov) {
        eye = Vec3f(rc * cos(phic), zc, -rc * sin(phic));
        gaze = Vec3f(rn * cos(phin), zn, -rn * sin(phin));
        pixel_width = px_w;
        pixel_height = px_h;
        field_of_view = fov * M_PI / 180;
    }

    Matrix44f cameraToWorld() const{
        Vec3f u, v, w, t = Vec3f(0,1,0);
        w = (eye - gaze).normalize();
        u = cross(t, w).normalize();
        v = cross(w, u);
        float m[4][4] = {{u.x, u.y, u.z, 0},
                         {v.x, v.y, v.z, 0},
                         {w.x, w.y, w.z, 0},
                         {eye.x, eye.y, eye.z, 1}};
        return Matrix44f(m);
    }
};

#endif //KP_TRACER_H
