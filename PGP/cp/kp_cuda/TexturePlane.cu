#include "TexturePlane.h"
#include <iostream>

using namespace std;

int my_sign(float num) {
    return (num < 0) ? -1 : (num > 0 ? 1 : 0);
}

TexturePlane::TexturePlane(float3 p[4], string texture_path, float scale, float3 color, float coef_ref)
: tex(texture_path, scale, color, coef_ref) {
    float3 ab = p[1] - p[0];
    float3 ac = p[2] - p[0];
    float3 ad = p[3] - p[0];

    //cout << "a: " << p[0] << endl;
    //cout << "b: " << p[1] << endl;
    //cout << "c: " << p[2] << endl;
    //cout << "d: " << p[3] << endl;

    //cout << "ab: " << ab << endl;
    //cout << "ac: " << ac << endl;
    //cout << "ad: " << ad << endl;
    float tmp = ad * cross(ab, ac);
    if (tmp  < 1e-6) {
        p0 = make_float3((p[0].x+p[1].x+p[2].x+p[3].x)*0.25,
                   (p[0].y+p[1].y+p[2].y+p[3].y)*0.25,
                   (p[0].z+p[1].z+p[2].z+p[3].z)*0.25);
        plane_normal = cross(ad, ab);
        plane_normal = normalize(plane_normal);
        //cout << "normal: " << plane_normal << endl;
        for (int i = 0; i <4; ++i) {
            this->p[i] = p[i];
        }
    }
    else{
        cerr << "Error: four points defining the floor are not coplanar!";
        exit(1);
    }
}

bool TexturePlane::check_point_inside(float3 pt) const{
    float3 n1 = cross(p[1]-pt, p[0] - pt);
    n1 = normalize(n1);
    if(!allclose(n1, plane_normal)) return false;

    float3 n2 = cross(p[2]-pt, p[1] - pt);
    n2 = normalize(n2);
    if(!allclose(n2, plane_normal))  return false;

    float3 n3 = cross(p[3]-pt, p[2] - pt);
    n3 = normalize(n3);
    if(!allclose(n3, plane_normal))   return false;
    float3 n4 = cross(p[0]-pt, p[3] - pt);
    n4 = normalize(n4);
    //cout << pt << " " << allclose(n1, plane_normal) << " " << allclose(n2, plane_normal) << " " << allclose(n3, plane_normal) << " " << allclose(n4, plane_normal) << endl;
    return allclose(n4, plane_normal);
}

TexturePlane::~TexturePlane() {}

CPUTexture::CPUTexture(string filename, float scale, float3 color, float coef_ref) {
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

float3 CPUTexture::GetPixel(float i, float j) const {
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

float3 CPUTexel::normalizedColor() {
    return make_float3(x/255.0, y/255.0, z/255.0);
}

__device__ bool GTexturePlane::check_point_inside(float3 pt) const {
    float3 n1 = normalize(cross(p1-pt, p0 - pt));
    if(!allclose(n1, plane_normal)) return false;
    float3 n2 = normalize(cross(p2-pt, p1 - pt));
    if(!allclose(n2, plane_normal))  return false;
    float3 n3 = normalize(cross(p3-pt, p2 - pt));
    if(!allclose(n3, plane_normal))   return false;
    float3 n4 = normalize(cross(p0-pt, p3 - pt));
    return allclose(n4, plane_normal);
}

__device__ bool GTexturePlane::hit(Ray ray, HitPoint &hp) {
    float denom = (-plane_normal) * ray.dir;
    if (denom > 1e-6) {
        float3 p0l0 = pc - ray.orig;
        hp.dist = p0l0 * (-plane_normal) * (1 / denom);
        hp.point = ray.get_point(hp.dist);
        if(check_point_inside(hp.point)) {
            float3 pp0 = (hp.point - pc) * scale_coef;
            //cout << ray << " " << t  << "===" << pt << endl;
            hp.N = plane_normal;
            float3 tmp = GetPixel(pp0.x, pp0.z);
            float3 tmp_color = tmp % color;
            hp.material = Material(1.0, make_float4(0.3,  1.0, 0.8, 0.0),
                    tmp_color,   100.);
            return (hp.dist >= 0);
        }
    }
    return false;
}

__device__ float3 GTexturePlane::GetPixel(float i, float j) {
    i = i - floor(i);
    j = j - floor(j);
    int ii = (int) (i * height);
    int jj = (int) (j * width);
    //printf("DATA: %d %d %d",data[ii*width + jj].x, data[ii*width + jj].y, data[ii*width + jj].z );
    //printf(" -- %f %f %f\n", normalizedColor( data[ii*width + jj]).x,normalizedColor( data[ii*width + jj]).y, normalizedColor( data[ii*width + jj]).z);
    return normalizedColor( data[ii*width + jj]);
}

GTexturePlane *createGTexturePlane(float3 *p, string texture_path, float scale, float3 color, float coef_ref) {
    GTexturePlane * pl;
    cudaMallocManaged(&pl, sizeof(GTexturePlane));
    pl->scale_coef = scale;
    pl->color = color;
    pl->coef_ref = coef_ref;

    float3 ab = p[1] - p[0];
    float3 ac = p[2] - p[0];
    float3 ad = p[3] - p[0];

    float tmp = ad * cross(ab, ac);
    if (tmp  < 1e-6) {
        pl->pc = make_float3((p[0].x + p[1].x + p[2].x + p[3].x) * 0.25,
                         (p[0].y+p[1].y+p[2].y+p[3].y)*0.25,
                         (p[0].z+p[1].z+p[2].z+p[3].z)*0.25);
        pl->plane_normal = normalize(cross(ad, ab));
        //cout << "normal: " << pl->plane_normal << endl;
        pl->p0 = p[0];
        pl->p1 = p[1];
        pl->p2 = p[2];
        pl->p3 = p[3];
    }
    else{
        cerr << "Error: four points defining the floor are not coplanar!";
        exit(1);
    }

    ifstream fin (texture_path, ios::in | ios::binary);
    if(!fin){
        cerr << "Can not load texture from file!\n";
        exit(1);
    }

    fin.read((char*)&pl->width, sizeof(unsigned int));
    fin.read((char*)&pl->height, sizeof(unsigned int));
    uchar4 *data_h = new uchar4[pl->width*pl->height];
    cudaMalloc(&pl->data, sizeof(uchar4)*pl->width*pl->height);
    fin.read((char*)data_h, sizeof(uchar4)*pl->width*pl->height);
    cudaMemcpy(pl->data, data_h, sizeof(uchar4)*pl->width*pl->height, cudaMemcpyHostToDevice);
    fin.close();


    return pl;
}
