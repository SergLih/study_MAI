#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>
#include "model.h"


float Model::spot_light_radius = 0.1;
Material Model::spot_light_material = Material(1.0, make_float4(1.0,  0.0, 0.0, 0.0), make_float3(1, 1, 1),   1.);
float Model::spot_light_intensity = 0.5;

// fills verts and faces arrays, supposes .obj file to have "f " entries without slashes
Model::Model(const char *filename, const Material &m, int lights_on_edge, float3 center, float radius) {
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
            float3 v;
            iss >> v;
            verts.push_back(v);
            if(thrust::find(lights_coords.begin(), lights_coords.end(), v) == lights_coords.end())
            {
                lights_coords.push_back(v);
                lights_normals.push_back(make_float3(0,0,0));
            }
            //cerr << "v: " << v << endl;
        } else if (!line.compare(0, 2, "f ")) {     //face -- грань, три номера вершин, которые её образуют
//            int3 f;
            int idx, cnt=0;
            iss >> trash;
            int t[3];
            while (iss >> idx) {
                t[cnt++] = --idx;  //нумерация вершин с нуля
            }
            if (3==cnt) {
                faces.push_back(make_int3(t[0], t[1], t[2]));
            }
//            cerr << "f: " << f.x << " " << f.y << " " << f.z << endl;
        }
    }
    for (int j = 0; j < faces.size(); ++j) {
        add_internal_edge_lights(j, lights_on_edge);
    }
    update_lights();
    //cerr << "#v=" << verts.size() << " #f=" << faces.size()
    //          << "#l=" << lights_coords.size() << " " << endl;   //отладочный вывод кол-ва вершин и граней

    this->scale(radius);
    this->translate(center);
}

float3 Model::point(int i) {
    assert(i>=0 && i<verts.size());
    return verts[i];
}

int Model::vert(int fi, int li) {
    assert(fi>=0 && fi<faces.size() && li>=0 && li<3);
    int3 t = faces[fi];
    if(li == 0)
        return t.x;
    else if(li == 1)
        return t.y;
    else
        return t.z;
}

float3 Model::get_normal(int fi) {
    float3 edge1 = point(vert(fi,1)) - point(vert(fi,0));  //сторона АВ = E1
    float3 edge2 = point(vert(fi,2)) - point(vert(fi,0));  //сторона AС = E2
    return normalize(cross(edge1, edge2));
}

void Model::translate(const float3 &delta) {
    for (int i = 0; i < verts.size(); ++i) {
        verts[i] = verts[i] + delta;
    }
    vector<float3> updated;
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
    vector<float3> updated;
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
        float3 start = verts[vert(fi, it_nv->first)],
                end = verts[vert(fi, it_nv->second)];
        for (int i = 1; i < k; ++i) {
            float t = i * 1.0 / (k - 1);
            float3 coord = start * t + end * (1 - t);
            float3 normal = get_normal(fi);
            auto it = thrust::find(lights_coords.begin(), lights_coords.end(), coord);
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
    lights_spheres.clear();
    for (auto it = lights_coords.begin(); it != lights_coords.end(); ++it) {
        lights_spheres.push_back(Sphere(*it, Model::spot_light_radius, Model::spot_light_material));
    }
}

Model::~Model() {}

GModel * Model::getGModel() {
    GModel *gm;
    cudaMallocManaged(&gm, sizeof(GModel));
    gm->n_verts  = this->verts.size();
    gm->n_faces  = this->faces.size();
    gm->n_lights = this->lights_coords.size();
    gm->material = this->material;
    cudaMalloc(&gm->faces, sizeof(int3)*gm->n_faces);
    cudaMemcpy(gm->faces, this->faces.data(), sizeof(int3)*gm->n_faces, cudaMemcpyHostToDevice);
    cudaMalloc(&gm->verts, sizeof(float3)*gm->n_verts);
    cudaMemcpy(gm->verts, this->verts.data(), sizeof(float3)*gm->n_verts, cudaMemcpyHostToDevice);

    cudaMalloc(&gm->light_spheres, sizeof(Sphere)*gm->n_lights);
    cudaMemcpy(gm->light_spheres, this->lights_spheres.data(), sizeof(Sphere)*gm->n_lights, cudaMemcpyHostToDevice);


    return gm;
}

void Model::freeGModel(GModel *gm) {
    cudaFree(gm->faces);
    cudaFree(gm->verts);
    cudaFree(gm);
}

__device__ float dmax(float v1, float v2) {
    return v1 > v2 ? v1 : v2;
}

__device__ float dmin(float v1, float v2) {
    return v1 < v2 ? v1 : v2;
}

//std::ostream& operator<<(std::ostream& out, const Ray & ray) {
//    out << "R[" << ray.orig << "]->[" << ray.dir << "]";
//    return out ;
//}

__device__ float3 refract(float3 I, float3 N, float refractive_index) { // Snell's law
    float cosi = - dmax(-1.f, dmin(1.f, I*N));
    float etai = 1, etat = refractive_index;
    float3 n = N;
    if (cosi < 0) { // if the ray is inside the object, swap the indices and invert the normal to get the correct result
        cosi = -cosi;
        n = -N;
        float tmp = etai;
        etai = etat;
        etat = tmp;
    }
    float eta = etai / etat;
    float k = 1 - eta*eta*(1 - cosi*cosi);
    return k < 0 ? make_float3(0,0,0) : I*eta + n*(eta * cosi - sqrtf(k));
}

__device__ float3 reflect(float3 I, float3 N) {
    return I - N*2.f*(I*N);
}

Material::Material(float coef_refr, float coef_refl, float3 color) {
    diffuse_color = color;
    refractive_index = coef_refr;
    if(coef_refl > 0.5){
        albedo = make_float4(0.0,  0.5, 0.1, 0.8);
        specular_exponent = coef_refl*2000;
    } else {
        albedo = make_float4(0.0, 10.0, 0.8, 0.0);
        specular_exponent = coef_refl*200;
    }
}

__host__ __device__ Material::Material(float r, float4 a, float3 color, float spec)   {
    refractive_index = r; albedo = a; diffuse_color = color;
    specular_exponent = spec;
}

__host__ __device__ Material::Material() {
    refractive_index = 1;
    albedo = make_float4(1, 0, 0, 0);
    diffuse_color = make_float3(1, 1, 1);
    specular_exponent = 1;
}

/*std::ostream & operator<<(std::ostream &ofs, const FrameBuffer &fb) {
    ofs << "P6\plane_normal" << fb.pixel_width << " " << fb.pixel_height << "\n255\plane_normal";
    for (size_t i = 0; i < fb.pixel_height*fb.pixel_width; ++i) {
        float3 c = fb.buf[i];
        float max = std::max(c.x, std::max(c.y, c.z));
        if (max > 1) c = c * (1. / max);
        for (size_t j = 0; j < 3; j++) {
            ofs << (char) (255 * std::max(0.f, std::min(1.f, fb.buf[i][j])));
        }
    }
    return ofs ;
}*/

bool operator<(const float3 &lhs, const float3 &rhs) {
    return (lhs.x < rhs.x || lhs.y < rhs.y || lhs.z < rhs.z);
}

bool operator==(const float3 &lhs, const float3 &rhs) {
    return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
}

bool operator!=(const float3 &lhs, const float3 &rhs) {
    return !(lhs == rhs);
}

float3 operator+(const float3 &lhs, const float3 &rhs) {
    return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

float3 operator-(const float3 &lhs, const float3 &rhs) {
    return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

float3 operator%(const float3 &lhs, const float3 &rhs) {
    return make_float3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}

float operator*(const float3 &lhs, const float3 &rhs) {
    return lhs.x * rhs.x +  lhs.y * rhs.y   + lhs.z * rhs.z;
}

float3 operator*(const float3 &lhs, float c) {
    return make_float3(lhs.x * c, lhs.y*c, lhs.z*c);
}

float3 operator-(const float3 &lhs) {
    return lhs * (-1);
}

float3 operator!(const float3 &lhs) {
    return make_float3(lhs.y, lhs.z, lhs.x);
}

float3 cross(const float3 &v1, const float3 &v2) {
    return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

bool allclose(const float3 &v1, const float3 &v2) {
    return abs(v1.x - v2.x) < v_eps && abs(v1.y - v2.y) < v_eps && abs(v1.z - v2.z) < v_eps;
}

std::ostream& operator<<(std::ostream& out, const float3 &v) {
    out << v.x << " " << v.y << " " << v.z;
    return out;
}

std::ofstream& operator<<(std::ofstream& out, const uchar3 &v) {
    out << v.x << v.y << v.z;
    return out;
}

std::istream& operator>>(std::istream& in, float3 &v) {
    in >> v.x >> v.y >> v.z;
    return in;
}

__host__ __device__ float norm(float3 v) { return sqrtf(v.x*v.x+v.y*v.y+v.z*v.z); }

__host__ __device__ float3 normalize(float3 v) { return norm(v)>0 ? v*(1.0f/norm(v)) : v; }

__host__ __device__ float clamp(float v) {
    if(v < 0) return 0;
    if(v > 1) return 1;
    return v;
}

__host__ __device__ uchar3 to_bytes(const float3 &v) {
    uchar3 res;
    float3 vcopy = v;
    float max = dmax(v.x, dmax(v.y, v.z));
    if (max > 1)
        vcopy = vcopy * (1. / max);
    res.x = (unsigned char) (255 * clamp(vcopy.x));
    res.y = (unsigned char) (255 * clamp(vcopy.y));
    res.z = (unsigned char) (255 * clamp(vcopy.z));
    return res;
}

__device__ bool Sphere::hit(const Ray &ray, HitPoint &hp) const {
    //функция находит коэф-т t по (единичному) вектору OD, возвр. true, если точка сущ.
    //https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
    float3 L = center - ray.orig;   //вектор L от начала к центру сферу
    float tca = L * ray.dir;         //расстояние до точки пересечения с перпендикуляром (проекция на единичный вектор). ее длина равна скалярному произведению.
    float d2 = L*L - tca*tca;  // квадрат расстояния от луча до центра сферы (перпендикуляр)
    if (d2 > radius*radius) return false; // если нет пересечения со сферой (расстояние до центра больше радиуса)
    float thc = sqrtf(radius*radius - d2);  //расстояние от точки пересечения сферы до точки пересечения с перпендикуляром (теорема пифагора)
    float t0       = tca - thc; //подсчет расстояния до первой и
    float t1 = tca + thc; //до второй т. пересеч. со сферой
    //см. рис. 3 https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
    if (t0 > t1) { float tmp = t1; t1 = t0; t0 = tmp; }
    if (t0 < 0)
        t0 = t1; // обработка случая когда начало луча внутри сферы
    if (t0 < 0)
        return false; // обработка случая когда сфера за лучом (угол за пределами 90 градусов)

    hp.dist = t0;
    hp.point = ray.get_point(t0);
    hp.N = normalize(ray.get_point(t0) - center);
    hp.material = this->material;
    return true;
}

float3 GModel::point(int i) const{
    assert(i>=0 && i<n_verts);
    return verts[i];
}

int GModel::vert(int fi, int li) const{
    assert(fi>=0 && fi<n_faces && li>=0 && li<3);
    int3 t = faces[fi];
    if(li == 0)
        return t.x;
    else if(li == 1)
        return t.y;
    else
        return t.z;
}

float3 GModel::get_normal(int fi) const {
    float3 edge1 = point(vert(fi, 1)) - point(vert(fi, 0));  //сторона АВ = E1
    float3 edge2 = point(vert(fi, 2)) - point(vert(fi, 0));  //сторона AС = E2
    return normalize(cross(edge1, edge2));
}

bool GModel::hit(Ray ray, HitPoint &hp) const {
    //printf("%f %f %f\plane_normal", ray.dir.x, ray.dir.y, ray.dir.z);
    //printf("hit\plane_normal");
    float t_i;
    int min_fi;
    float min_t = 1e38; //сначало берем максимальное число,
    bool found = false;
    // для каждого тр-ка модели считаем расстояние (через параметр t) от нашей камеры до него, ищем минимум))
    for (int fi = 0; fi < n_faces; ++fi) {
        if(ray_triangle_intersect(fi, ray, t_i)) {
            //printf("SSS: %f %f\plane_normal", t_i, min_t);
            if (t_i < min_t) {
                min_t = t_i;
                min_fi = fi;
                found = true;
            }
        }
    }
    if(found){
        //printf("HIT\plane_normal");
        hp.dist = min_t;
        hp.point = ray.get_point(min_t);
        hp.N = get_normal(min_fi);
        hp.material = this->material;
    }
    //printf("%f\plane_normal", min_t);
    return found;
}

// Moller and Trumbore
bool GModel::ray_triangle_intersect(const int &fi, const Ray &ray, float &t) const {
    //printf("rt_intersect\plane_normal");
    //printf("%f %f %f\t EDGES: ", ray.dir.x, ray.dir.y, ray.dir.z);
    float eps = 1e-5;
    float3 edge1 = point(vert(fi,1)) - point(vert(fi,0));  //сторона АВ = E1
    float3 edge2 = point(vert(fi,2)) - point(vert(fi,0));  //сторона AС = E2
    //printf("%f %f %f\t", edge1.x, edge1.y, edge1.z);
    //printf("%f %f %f\plane_normal", edge2.x, edge2.y, edge2.z);
    float3 pvec = cross(ray.dir, edge2);     //(D x E2) = P, векторное произведение двух векторов дает нормаль которая перпендикулярна плоскости, которую любые два вектора определяют

    float det = edge1*pvec;             //det = (D x E2) . E1 = P . E1  определитель (число) в знамен. метода Крамера
    //printf("DET = %f\plane_normal", det);
    if (det<eps)                        //если он близок к нулю (|det| < eps) то луч параллелен плоск-ти тр-ка
        return false;                   //если он меньше нуля (det < -eps) то back culling (не рисуем тр-ки, у которых нормаль повернута туда же куда и луч камеры

    float3 tvec = ray.orig - point(vert(fi,0));  //T = O - A
    float u = pvec*tvec* (1./det);               //бариц. координата u = [ (D x E2) . T ] / det= ( P . T )/ det
    if (u < 0 || u > 1)                          //если выходит за пределы [0, 1], то точка пересечения лежит за пределами тр-ка
        return false;

    float3 qvec = cross(tvec, edge1);            //Q = T x E1
    //printf("Q = %f %f %f\plane_normal", qvec.x, qvec.y, qvec.z);
    float v = qvec*ray.dir* (1./det);               //бариц. координата v = [ (T x E1) . D ] / det= ( Q . D )/ det
    if (v < 0 || u + v > 1)                     //если выходит за пределы [0, 1], то точка пересечения лежит за пределами тр-ка
        return false;

    t = edge2*qvec * (1./det);                  //третья координата (см. рис. 1 в https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection)
    //printf("TT = %f\plane_normal", t);
    return t > eps;                               //возвращаем true, если точка пересечения находится перед, а не за, или совсем рядом с камерой.
}

float3 normalizedColor(uchar4 v) {
    return make_float3(v.x/255.0, v.y/255.0, v.z/255.0);
}

__device__ float3 Ray::get_point(float t) const {
    return orig + dir * t;
}

__device__  Ray::Ray(float3 orig, float3 dir, float coef, unsigned short i, unsigned short j, unsigned char depth) {
    this->orig  = orig;
    this->dir   = dir;
    this->depth = depth;
    this->coef  = coef;
    this->i     = i;
    this->j     = j;
}

