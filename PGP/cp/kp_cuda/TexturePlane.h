#ifndef KP_TEXTUREPLANE_H
#define KP_TEXTUREPLANE_H

#include "model.h"
#include <fstream>
#include <cmath>

using namespace std;
typedef unsigned char uchar;

struct CPUTexel{
    uchar x;
    uchar y;
    uchar z;
    uchar w;

    float3 normalizedColor();
};

class CPUTexture{
private:
    unsigned int width;
    unsigned int height;
    CPUTexel * data;

    static inline void endian_swap(unsigned int& x);
public:
    float scale_coef;
    float3 color;
    float coef_ref;
    CPUTexture(string filename, float scale, float3 color, float coef_ref);
    ~CPUTexture();
    float3 GetPixel(float i, float j) const;

};

class TexturePlane {
private:
    float3 p0;
    float3 plane_normal;

    float3 p[4];

    CPUTexture tex;

    bool check_point_inside(float3 pt) const;

public:
    TexturePlane(float3 p[4], string texture_path, float scale, float3 color, float coef_ref);
    ~TexturePlane() ;
    bool hit(const Ray &ray, float &t, float3 &normal, Material &material);
};

struct GTexturePlane {
public:
    float3 pc;
    float3 plane_normal;

    float3 p0, p1, p2, p3;

    float scale_coef;
    float3 color;
    float coef_ref;

    unsigned int width;
    unsigned int height;

    uchar4 * data;

    __device__ bool check_point_inside(float3 pt) const;

    __device__ bool hit(Ray ray, HitPoint &hp);
    __device__ float3 GetPixel(float i, float j);
};

__host__ GTexturePlane * createGTexturePlane(float3 *p, string texture_path, float scale, float3 color, float coef_ref);
#endif //KP_TEXTUREPLANE_H
