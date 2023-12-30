#ifndef KP_SPHERE_H
#define KP_SPHERE_H

#include "tracer.h"

class Sphere : public IHittable {
private:
    Vec3f center;
    float radius;
    Material material;

public:
    Sphere(const Vec3f &c, const float &r, const Material &m) : center(c), radius(r), material(m) {}
    ~Sphere();
    bool hit(const Ray &ray, float &t, Vec3f &normal, Material &material) const override;
};

#endif //KP_SPHERE_H
