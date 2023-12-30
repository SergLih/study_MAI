#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>
#include "model.h"
#include "sphere.h"

float Model::spot_light_radius = 0.1;
Material Model::spot_light_material = Material(1.0, Vec4f(1.0,  0.0, 0.0, 0.0), Vec3f(1, 1, 1),   1.);
float Model::spot_light_intensity = 0.5;

// fills verts and faces arrays, supposes .obj file to have "f " entries without slashes
Model::Model(const char *filename, const Material &m, int lights_on_edge, Vec3f center, float radius) {
    material = m;
    ifstream in;
    in.open (filename, ifstream::in);
    if (in.fail()) {
        cerr << "Failed to open " << filename << endl;
        return;
    }
    string line;
    while (!in.eof()) {
        getline(in, line);
        istringstream iss(line.c_str());
        char trash;
        if (!line.compare(0, 2, "v ")) {            //vertex -- вершина, три координаты
            iss >> trash;
            Vec3f v;
            for (int i=0;i<3;i++) iss >> v[i];
            verts.push_back(v);
            auto it = find(lights_coords.begin(), lights_coords.end(), v);
            if(it == lights_coords.end())
            {
                lights_coords.push_back(v);
                lights_normals.push_back(Vec3f());
            }
            //cerr << "v: " << v << endl;
        } else if (!line.compare(0, 2, "f ")) {     //face -- грань, три номера вершин, которые её образуют
            Vec3i f;
            int idx, cnt=0;
            iss >> trash;
            while (iss >> idx) {
                idx--;                                          //нумерация вершин с нуля
                f[cnt++] = idx;
            }
            if (3==cnt) faces.push_back(f);
            //cerr << "f: " << f << endl;
        }
    }
    for (int j = 0; j < nfaces(); ++j) {
        add_internal_edge_lights(j, lights_on_edge);
    }
    //update_lights();
    //cerr << "#v=" << verts.size() << " #f=" << faces.size()
    //          << "#l=" << lights_coords.size() << " " << endl;   //отладочный вывод кол-ва вершин и граней

    Vec3f min, max;
    this->scale(radius);
    this->translate(center);
}

// Moller and Trumbore
bool Model::ray_triangle_intersect(const int &fi, const Vec3f &orig, const Vec3f &dir, float &t) const {
    float eps = 1e-5;
    Vec3f edge1 = point(vert(fi,1)) - point(vert(fi,0));  //сторона АВ = E1
    Vec3f edge2 = point(vert(fi,2)) - point(vert(fi,0));  //сторона AС = E2
    Vec3f pvec = cross(dir, edge2);     //(D x E2) = P, векторное произведение двух векторов дает нормаль которая перпендикулярна плоскости, которую любые два вектора определяют
    float det = edge1*pvec;             //det = (D x E2) . E1 = P . E1  определитель (число) в знамен. метода Крамера
    if (det<eps)                        //если он близок к нулю (|det| < eps) то луч параллелен плоск-ти тр-ка
        return false;                   //если он меньше нуля (det < -eps) то back culling (не рисуем тр-ки, у которых нормаль повернута туда же куда и луч камеры

    Vec3f tvec = orig - point(vert(fi,0));  //T = O - A
    float u = pvec*tvec* (1./det);               //бариц. координата u = [ (D x E2) . T ] / det= ( P . T )/ det
    if (u < 0 || u > 1)                          //если выходит за пределы [0, 1], то точка пересечения лежит за пределами тр-ка
        return false;

    Vec3f qvec = cross(tvec, edge1);            //Q = T x E1
    float v = qvec*dir* (1./det);               //бариц. координата v = [ (T x E1) . D ] / det= ( Q . D )/ det
    if (v < 0 || u + v > 1)                     //если выходит за пределы [0, 1], то точка пересечения лежит за пределами тр-ка
        return false;

    t = edge2*qvec * (1./det);                  //третья координата (см. рис. 1 в https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection)
                                                //t  = [ (T x E1) . E2 ] / det= ( Q . E2 )/ det
    return t>eps;                               //возвращаем true, если точка пересечения находится перед, а не за, или совсем рядом с камерой.
}


int Model::nverts() const {
    return (int)verts.size();
}

int Model::nfaces() const {
    return (int)faces.size();
}

const Vec3f &Model::point(int i) const {
    assert(i>=0 && i<nverts());
    return verts[i];
}

Vec3f &Model::point(int i) {
    assert(i>=0 && i<nverts());
    return verts[i];
}

int Model::vert(int fi, int li) const {
    assert(fi>=0 && fi<nfaces() && li>=0 && li<3);
    return faces[fi][li];
}

Vec3f Model::get_normal(int fi) const {
    Vec3f edge1 = point(vert(fi,1)) - point(vert(fi,0));  //сторона АВ = E1
    Vec3f edge2 = point(vert(fi,2)) - point(vert(fi,0));  //сторона AС = E2
    return cross(edge1, edge2).normalize();
}

bool Model::hit(const Ray &ray, float &min_t, Vec3f &normal, Material &material) const {
    float t_i;
    int min_fi;
    min_t = numeric_limits<float>::max(); //сначало берем максимальное число,
    bool found = false;
    // для каждого тр-ка модели считаем расстояние (через параметр t) от нашей камеры до него, ищем минимум
    for (int fi = 0; fi < nfaces(); ++fi) {
        if(ray_triangle_intersect(fi, ray.orig, ray.dir, t_i))
            if(t_i < min_t){
                min_t = t_i;
                min_fi = fi;
                found = true;
            }
    }
    if(found){
        normal = get_normal(min_fi);
        material = this->material;
    }
    return found;
}

void Model::translate(const Vec3f &delta) {
    for (int i = 0; i < verts.size(); ++i) {
        verts[i] = verts[i] + delta;
    }
    vector<Vec3f> updated;
    for (auto it = lights_coords.begin(); it != lights_coords.end(); ++it) {
        updated.push_back(*it + delta);
    }
    lights_coords = updated;
    update_lights();
}

void Model::scale(float coef) {
    for (int i = 0; i < verts.size(); ++i) {
        verts[i] = verts[i] * coef;
    }
    vector<Vec3f> updated;
    for (auto it = lights_coords.begin(); it != lights_coords.end(); ++it) {
        updated.push_back(*it * coef);
    }
    lights_coords = updated;
    update_lights();
}

void Model::add_internal_edge_lights(int fi, int lights_on_edge) {
    int k = lights_on_edge;
    vector<pair<int, int>> nv = {{0, 1}, {1, 2}, {2, 0}};
    for(auto it_nv = nv.begin(); it_nv != nv.end(); it_nv++) {
        Vec3f start = verts[vert(fi, it_nv->first)],
                end = verts[vert(fi, it_nv->second)];
        for (int i = 1; i < k; ++i) {
            float t = i * 1.0 / (k - 1);
            Vec3f coord = start * t + end * (1 - t);
            Vec3f normal = get_normal(fi);
            auto it = find(lights_coords.begin(), lights_coords.end(), coord);
            int j = it - lights_coords.begin();
            if(it != lights_coords.end() && lights_normals[j] == normal) {
                lights_coords.erase(it);
                lights_normals.erase(lights_normals.begin()+j);
            } else {
                lights_coords.push_back(coord);
                lights_normals.push_back(normal);
            }
        }
    }
}

void Model::update_lights() {
    //lights.clear();
    for (int i = 0; i < lights_spheres.size(); ++i) {
        delete lights_spheres[i];
    }
    lights_spheres.clear();
    for (auto it = lights_coords.begin(); it != lights_coords.end(); ++it) {
        //lights.push_back(Light(*it , Vec3f(1, 1, 1), Model::spot_light_intensity));
        lights_spheres.push_back(new Sphere(*it, Model::spot_light_radius, Model::spot_light_material));
    }
}

Model::~Model() {}

ostream& operator<<(ostream& out, Model &m) {
    // out << "#v = " << m.nverts() << ", #f = "  << m.nfaces() << endl;
    // for (int i=0; i<m.nverts(); i++) {
    //     out << "v: " << m.point(i) << endl;
    // }
//    for (int i=0; i<m.nfaces(); i++) {
//        out << "f: ";
//        for (int k=0; k<3; k++) {
//            out << (m.vert(i,k)+1) << " ";
//        }
//        out << endl;
//    }
    return out;
}

