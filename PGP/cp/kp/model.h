#ifndef __MODEL_H__
#define __MODEL_H__
#include <vector>
#include <string>
#include <unordered_set>
#include "vector.hpp"
#include "tracer.h"

class Sphere;

using namespace std;

class Model : public IHittable {
private:
    vector<Vec3f> verts;
    vector<Vec3i> faces;
    vector<Vec3f> lights_coords;
    vector<Vec3f> lights_normals;


    //поиск пересечение луча с fi-й гранью
    bool ray_triangle_intersect(const int &fi, const Vec3f &orig, const Vec3f &dir, float &t) const;
    void add_internal_edge_lights(int fi, int lights_on_edge);
    void update_lights();
public:
    Material material;
    vector<Sphere*> lights_spheres;

    Model(const char *filename, const Material & m, int lights_on_edge, Vec3f center=Vec3f(0, 0, 0), float radius=1);
    ~Model() override;

    int nverts() const;                          // число вершин
    int nfaces() const;                          // число граней - треугольников

    bool hit(const Ray &ray, float &t, Vec3f &normal, Material &material) const override;

    const Vec3f &point(int i) const;             // получить координаты i-й точки (конст.)
    Vec3f &point(int i);                         // получить координаты i-й точки (доступно для изменения)
    int vert(int fi, int li) const;              // получить координаты li-й точки (нум.с 0) fi-й грани
    Vec3f get_normal(int fi) const;                    //получить нормаль к fi-й грани
    void translate(const Vec3f &delta);
    void scale(float coef);

    static float spot_light_radius;
    static Material spot_light_material;
    static float spot_light_intensity;
};

ostream& operator<<(ostream& out, Model &m);     //вывод на экран / в файл

#endif //__MODEL_H__

