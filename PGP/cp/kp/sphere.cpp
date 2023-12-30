#include "sphere.h"

bool Sphere::hit(const Ray &ray, float &t0, Vec3f &normal, Material &material) const {
    //функция находит коэф-т t по (единичному) вектору OD, возвр. true, если точка сущ.
    //https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
    Vec3f L = center - ray.orig;   //вектор L от начала к центру сферу
    float tca = L * ray.dir;         //расстояние до точки пересечения с перпендикуляром (проекция на единичный вектор). ее длина равна скалярному произведению.
    float d2 = L*L - tca*tca;  // квадрат расстояния от луча до центра сферы (перпендикуляр)
    if (d2 > radius*radius) return false; // если нет пересечения со сферой (расстояние до центра больше радиуса)
    float thc = sqrtf(radius*radius - d2);  //расстояние от точки пересечения сферы до точки пересечения с перпендикуляром (теорема пифагора)
    t0       = tca - thc; //подсчет расстояния до первой и
    float t1 = tca + thc; //до второй т. пересеч. со сферой
    //см. рис. 3 https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
    if (t0 > t1) std::swap(t0, t1);
    if (t0 < 0)
        t0 = t1; //обработка случая когда начало луча внутри сферы
     if (t0 < 0)
         return false; //обработка случая когда сфера за лучом (угол за пределами 90 градусов)

    normal = (ray.get_point(t0) - center).normalize();
    material = this->material;
    return true;
}

Sphere::~Sphere() { }

