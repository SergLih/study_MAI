#include <limits>
#include "tracer.h"
#include "sphere.h"
#include "TexturePlane.h"

//std::ostream& operator<<(std::ostream& out, const Ray & ray) {
//    out << " Ray [" << ray.orig << " -> " << ray.dir << "] ";
//    return out ;
//}

IHittable::~IHittable() {}

int Ray::max_depth;
int Ray::rays_per_pixel;

Vec3f Ray::cast(const std::vector<IHittable*> &objects, const std::vector<Light> &lights, size_t depth) {
    Vec3f point, N;         //выходные пар-ры функции scene_intersect
    IHittable * hit_obj, *shadow_hit_obj;
    Material material;
    if (depth>Ray::max_depth || !scene_intersect(objects, hit_obj, point, N, material)) {
        return Vec3f(0.0, 0.0, 0.0); // background color
    }

    if(typeid(*hit_obj) == typeid(Sphere))
        return Vec3f(1,1,1);

//    if(typeid(*hit_obj)== typeid(TexturePlane)) {
//        cout << N << "\n";
//    }

    Vec3f diffuse_light_intensity, specular_light_intensity;
    for (size_t i=0; i<lights.size(); i++) { //по всем источникам света
        Vec3f light_dir      = (lights[i].position - point).normalize();

        float light_distance = (lights[i].position - point).norm();


            Vec3f shadow_orig = light_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3; // checking if the point lies in the shadow of the lights_coords[i]
            Vec3f shadow_pt, shadow_N;
            Material tmpmaterial;
            Ray tmp_ray(shadow_orig, light_dir);
            if (tmp_ray.scene_intersect(objects, shadow_hit_obj, shadow_pt, shadow_N, tmpmaterial)
                && typeid(*shadow_hit_obj) != typeid(Sphere)
                && (shadow_pt - shadow_orig).norm() < light_distance)
                continue;


        diffuse_light_intensity  = diffuse_light_intensity +
                (material.diffuse_color % lights[i].color) * lights[i].intensity *
                std::max(0.f, light_dir*N);
        specular_light_intensity = specular_light_intensity +
                (Vec3f(1., 1., 1.) % lights[i].color) *
                powf(std::max(0.f, -reflect(-light_dir, N)*this->dir), material.specular_exponent) *
                lights[i].intensity;

//        if(typeid(*hit_obj)== typeid(TexturePlane)) {
//            cout << "[" << i << "]" << material.diffuse_color % lights[i].color << "|"
//            <<light_dir*N << "\n";
//        }
    }
    Vec3f result_color = diffuse_light_intensity * material.albedo[0]
                         + specular_light_intensity * material.albedo[1];

    //cout << "C: " + to_string(result_color.x) + " " + to_string(result_color.y) + " "
    //        + to_string(result_color.z) + "\n";

    if(typeid(*hit_obj) != typeid(Sphere)) {
        Vec3f reflect_dir = reflect(this->dir, N).normalize();
        Vec3f refract_dir = refract(this->dir, N, material.refractive_index).normalize();
        Vec3f reflect_orig = reflect_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3; // offset the original point to avoid occlusion by the object itself
        Vec3f refract_orig = refract_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3;
        Ray reflect_ray(reflect_orig, reflect_dir);
        Vec3f reflect_color = reflect_ray.cast(objects, lights, depth + 1);
        Ray refract_ray(refract_orig, refract_dir);
        Vec3f refract_color = refract_ray.cast(objects, lights, depth + 1);
        result_color = result_color + reflect_color*material.albedo[2]
                                    + refract_color*material.albedo[3];

    }
    return result_color;
}

bool Ray::scene_intersect(const std::vector<IHittable*> &objects, IHittable* &hit_obj, Vec3f &hitpoint, Vec3f &normal, Material &material) {
    float cur_dist, min_dist = std::numeric_limits<float>::max(); //сначало берем максимальное число, для каждой сферы считаем расстояние от нашей камеры до сферы
    Vec3f cur_normal;
    Material cur_material;
    for (int i = 0; i < objects.size(); ++i) {
        if (objects[i]->hit(*this, cur_dist, cur_normal, cur_material) && cur_dist < min_dist) { //чтобы найти грань модели, которая на переднем плане
            min_dist = cur_dist; // и запоминаем у этой сферы точку пересечения
            hitpoint = this->get_point(cur_dist);
            hit_obj = objects[i];
            normal = cur_normal;
            material = cur_material;
        }
    }
    return min_dist < std::numeric_limits<float>::max();       // true, если хоть одна сфера пересекает луч
}

Ray::Ray() {}

std::ostream& operator<<(std::ostream& out, const Ray & ray) {
    out << "R[" << ray.orig << "]->[" << ray.dir << "]";
    return out ;
}

Vec3f refract(const Vec3f &I, const Vec3f &N, const float &refractive_index) { // Закон Снеллиуса
    float cosi = - std::max(-1.f, std::min(1.f, I*N));
    float etai = 1, etat = refractive_index;
    Vec3f n = N;
    if (cosi < 0) {
        cosi = -cosi;
        std::swap(etai, etat); n = -N;
    }
    float eta = etai / etat;
    float k = 1 - eta*eta*(1 - cosi*cosi);
    return k < 0 ? Vec3f(0,0,0) : I*eta + n*(eta * cosi - sqrtf(k));
}

Vec3f reflect(const Vec3f &I, const Vec3f &N) {
    return I - N*2.f*(I*N);
}

