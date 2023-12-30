#include "TexturePlane.h"

int my_sign(float num) {
    return (num < 0) ? -1 : (num > 0 ? 1 : 0);
}

TexturePlane::TexturePlane(Vec3f p[4], string texture_path, float scale, Vec3f color, float coef_ref)
: tex(texture_path, scale, color, coef_ref) {
    Vec3f ab = p[1] - p[0];
    Vec3f ac = p[2] - p[0];
    Vec3f ad = p[3] - p[0];

    // cout << "a: " << p[0] << endl;
    // cout << "b: " << p[1] << endl;
    // cout << "c: " << p[2] << endl;
    // cout << "d: " << p[3] << endl;

    // cout << "ab: " << ab << endl;
    // cout << "ac: " << ac << endl;
    // cout << "ad: " << ad << endl;
    float tmp = ad * cross(ab, ac);
    if (tmp  < 1e-6) {
        p0 = Vec3f((p[0].x+p[1].x+p[2].x+p[3].x)*0.25,
                   (p[0].y+p[1].y+p[2].y+p[3].y)*0.25,
                   (p[0].z+p[1].z+p[2].z+p[3].z)*0.25);
        n = cross(ad, ab).normalize();
        //cout << "normal: " << n << endl;
        for (int i = 0; i <4; ++i) {
            this->p[i] = p[i];
        }
    }
    else{
        cerr << "Error: four points defining the floor are not coplanar!";
        exit(1);
    }
}

bool TexturePlane::hit(const Ray &ray, float &t, Vec3f &normal, Material &material) const {
    float denom = (-n) * ray.dir;
    if (denom > 1e-6) {
        Vec3f p0l0 = p0 - ray.orig;
        t = p0l0 * (-n) * (1/denom);
        //if(t > 0 && t < 50)
            //cout << "*";
        Vec3f pt = ray.get_point(t);
        if(check_point_inside(pt)) {
            Vec3f pp0 = (pt - p0)*tex.scale_coef;
            //cout << ray << " " << t  << "===" << pt << endl;
            normal = n;
            Vec3f tmp = tex.GetPixel(pp0.x, pp0.z);
            Vec3f color = tmp % tex.color;
            material = Material(1.0, Vec4f(0.3,  1.0, 0.8, 0.0),
                    color,   100.);
            return (t >= 0);
        }
        else
            return false;
    }

    return false;
}


bool TexturePlane::check_point_inside(Vec3f pt) const{
    Vec3f n1 = cross(p[1]-pt, p[0] - pt).normalize();
    if(!allclose(n1, n)) return false;

    Vec3f n2 = cross(p[2]-pt, p[1] - pt).normalize();
    if(!allclose(n2, n))  return false;

    Vec3f n3 = cross(p[3]-pt, p[2] - pt).normalize();
    if(!allclose(n3, n))   return false;
    Vec3f n4 = cross(p[0]-pt, p[3] - pt).normalize();
    //cout << pt << " " << allclose(n1, n) << " " << allclose(n2, n) << " " << allclose(n3, n) << " " << allclose(n4, n) << endl;
    return allclose(n4, n);
}

TexturePlane::~TexturePlane() {}

CPUTexture::CPUTexture(string filename, float scale, Vec3f color, float coef_ref) {
    scale_coef = scale;
    this->color = color;
    this->coef_ref = coef_ref;
    ifstream fin (filename, ios::in | ios::binary);
    if(!fin){
        cerr << "Can not load texture from file!\n";
        exit(1);
    }
    fin.read((char*)&width, sizeof(unsigned int));
    fin.read((char*)&height, sizeof(unsigned int));
    data = new CPUTexel[width * height];
    int i = 0;
    while(!fin.eof() && i < width * height)
        fin.read((char*)&data[i++], sizeof(CPUTexel));
    fin.close();
}

Vec3f CPUTexture::GetPixel(float i, float j) const {
    i = i - floor(i);
    j = j - floor(j);
    int ii = (int) (i * height);
    int jj = (int) (j * width);
    return data[ii*width + jj].normalizedColor();
}

void CPUTexture::endian_swap(unsigned int &x) {
    x = (x>>24) |
        ((x<<8) & 0x00FF0000) |
        ((x>>8) & 0x0000FF00) |
        (x<<24);
}

CPUTexture::~CPUTexture() {
    delete[] data;
}

Vec3f CPUTexel::normalizedColor() {
    return Vec3f(x/255.0, y/255.0, z/255.0);
}
