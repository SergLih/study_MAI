#ifndef KP_TEXTUREPLANE_H
#define KP_TEXTUREPLANE_H


#include "tracer.h"
#include <fstream>
#include <cmath>

using namespace std;
typedef unsigned char uchar;

struct CPUTexel{
    uchar x;
    uchar y;
    uchar z;
    uchar w;

    Vec3f normalizedColor();
};

class CPUTexture{
private:
    unsigned int width;
    unsigned int height;
    CPUTexel * data;

    static inline void endian_swap(unsigned int& x);
public:
    float scale_coef;
    Vec3f color;
    float coef_ref;
    CPUTexture(string filename, float scale, Vec3f color, float coef_ref);
    ~CPUTexture();
    Vec3f GetPixel(float i, float j) const;

};


class TexturePlane : public IHittable {
private:
    Vec3f p0;
    Vec3f n;

    Vec3f p[4];

    CPUTexture tex;

    bool check_point_inside(Vec3f pt) const;

public:
    TexturePlane(Vec3f p[4], string texture_path, float scale, Vec3f color, float coef_ref);
    ~TexturePlane() override ;
    bool hit(const Ray &ray, float &t, Vec3f &normal, Material &material) const override;

};


#endif //KP_TEXTUREPLANE_H
