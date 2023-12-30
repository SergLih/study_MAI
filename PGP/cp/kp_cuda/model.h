#include <vector>
#include <string>
#include <unordered_set>
#include <array>
#include <thrust/device_vector.h>
#include <thrust/find.h>
#include <vector_types.h>

#ifndef __MODEL_H__
#define __MODEL_H__

#include "Matrix.h"

__host__ __device__ float  dmax(float v1, float v2);
__host__ __device__ float  dmin(float v1, float v2);

struct Ray{
    float3 orig;
    float3 dir;
    float coef;
    unsigned short i;
    unsigned short j;
    unsigned char depth;

    __host__ __device__ Ray(float3 orig, float3 dir, float coef, unsigned short i, unsigned short j, unsigned char depth=1);
    __device__ float3 get_point(float t) const;

};

class Model;

struct Material {

    __host__ __device__ Material(float r, float4 a, float3 color,
            float spec);
    __host__ __device__ Material();
    __host__ __device__ Material(float coef_refr, float coef_refl, float3 color);
    float refractive_index;
    float4 albedo;
    float3 diffuse_color;
    float specular_exponent;
};

struct HitPoint {
    float dist;
    float3 point;
    float3 N;
    Material material;
};


class GModel;


class Sphere {
private:
    float3 center;
    float radius;
    Material material;

public:
    Sphere(float3 c, float r, Material m) : center(c), radius(r), material(m) {}
    __device__ bool hit(const Ray &ray, HitPoint &hp) const;
};

using namespace std;

enum class IHittableType {
    Sphere, TexturePlane, Model
};

__device__ float3 reflect(float3 I, float3 N);
__device__ float3 refract(float3 I, float3 N, float refractive_index);
std::ostream& operator<<(std::ostream& out, const Ray & r);



class GModel {
public:
    float3 * verts;
    int n_verts;
    int3 * faces;
    int n_faces;
    int n_lights;
    Material material;
    Sphere * light_spheres;

    __device__ bool ray_triangle_intersect(const int &fi, const Ray &ray, float &t) const;
    __device__ float3 point(int i) const;                   // получить координаты i-й точки (доступно для изменения)
    __device__ int vert(int fi, int li) const;              // получить координаты li-й точки (нум.с 0) fi-й грани
    __device__ float3 get_normal(int fi) const;             //получить нормаль к fi-й грани

    __device__ bool hit(Ray ray, HitPoint &hp) const;
};

class Model {

private:
    vector<float3> verts;
    vector<int3> faces;
    vector<float3> lights_coords;
    vector<float3> lights_normals;

    //функция, ищущая пересечение луча с fi-й гранью
    void add_internal_edge_lights(int fi, int lights_on_edge);
    void update_lights();
    void translate(const float3 &delta);
    void scale(float coef);
public:
    Material material;
    vector<Sphere> lights_spheres;

    Model(const char *filename, const Material & m, int lights_on_edge, float3 center=make_float3(0, 0, 0), float radius=1);
    ~Model();

    GModel * getGModel();

    static void freeGModel(GModel *gm);

    float3 point(int i);                   // получить координаты i-й точки (доступно для изменения)
    int vert(int fi, int li);              // получить координаты li-й точки (нум.с 0) fi-й грани
    float3 get_normal(int fi);             //получить нормаль к fi-й грани


    static float spot_light_radius;
    static Material spot_light_material;
    static float spot_light_intensity;
};

//ostream& operator<<(ostream& out, Model &m);

const float v_eps = 1e-3;
__host__ __device__ float norm(float3 v);
__host__ __device__ float3 normalize(float3 v);
__host__ __device__ float3 operator% (const float3 &lhs, const float3 &rhs);
__host__ __device__ bool operator<(const float3 &lhs, const float3 &rhs);
__host__ __device__ bool operator==(const float3 &lhs, const float3 & rhs);
__host__ __device__ bool operator!=(const float3 &lhs, const float3 & rhs);
__host__ __device__ float3 operator+(const float3 &lhs, const float3 & rhs);
__host__ __device__ float3 operator-(const float3 &lhs, const float3 & rhs);
__host__ __device__ float operator*(const float3 &lhs, const float3 & rhs);
__host__ __device__ float3 operator*(const float3 &lhs, float c);
__host__ __device__ float3 operator-(const float3 &lhs);
__host__ __device__ float3 operator!(const float3 &lhs);
__host__ __device__ float3 cross(const float3 &v1, const float3 &v2);
__host__ __device__ bool allclose(const float3 &v1, const float3 &v2);
__host__ __device__ std::ostream& operator<<(std::ostream& out, const float3 &v);
__host__ __device__ std::ofstream& operator<<(std::ofstream& out, const uchar3 &v);
__host__ __device__ uchar3 to_bytes(const float3 &v);
__host__ __device__ float clamp(float v);
__host__ __device__ std::istream& operator>>(std::istream& in, float3 &v);
__host__ __device__ float3 normalizedColor(uchar4 v);

#endif //__MODEL_H__

