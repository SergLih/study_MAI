#ifndef KP_MAfloatRIX_HPP
#define KP_MAfloatRIX_HPP

#include <cstdint>
#include <iomanip>
#include <iostream>


class Matrix44 {
public:

    float m[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    static const Matrix44 kIdentity;

    __host__ __device__ Matrix44() {}

    __host__ __device__ Matrix44(float data[4][4]) {
        for (uint8_t i = 0; i < 4; ++i) {
            for (uint8_t j = 0; j < 4; ++j) {
                m[i][j] = data[i][j];
            }
        }
    }

    // Умножить текущую матрицу на другую (справа)
    __device__ Matrix44 operator * (const Matrix44& v) const {
        Matrix44 tmp;
        multiply(*this, v, tmp);

        return tmp;
    }

    __device__ static void multiply(const Matrix44 &a, const Matrix44& b, Matrix44 &c) {
        for (uint8_t i = 0; i < 4; ++i) {
            for (uint8_t j = 0; j < 4; ++j) {
                c.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] +
                          a.m[i][2] * b.m[2][j] + a.m[i][3] * b.m[3][j];
            }
        }
    }

    __device__ Matrix44 transposed() const {
        Matrix44 t;
        for (uint8_t i = 0; i < 4; ++i) {
            for (uint8_t j = 0; j < 4; ++j) {
                t.m[i][j] = m[j][i];
            }
        }
        return t;
    }

    __device__ void multVecMatrix(const float3 &src, float3 &dst) const {
        float a, b, c, w;
        a = src.x * m[0][0] + src.y * m[1][0] + src.z * m[2][0] + m[3][0];
        b = src.x * m[0][1] + src.y * m[1][1] + src.z * m[2][1] + m[3][1];
        c = src.x * m[0][2] + src.y * m[1][2] + src.z * m[2][2] + m[3][2];
        w = src.x * m[0][3] + src.y * m[1][3] + src.z * m[2][3] + m[3][3];

        dst.x = a / w;
        dst.y = b / w;
        dst.z = c / w;
    }

    //не учитываем коэф-ты отвечающие за сдвиг (translation)
    __host__ __device__ void multDirMatrix(const float3 &src, float3 &dst) const {
        dst.x = src.x * m[0][0] + src.y * m[1][0] + src.z * m[2][0];
        dst.y = src.x * m[0][1] + src.y * m[1][1] + src.z * m[2][1];
        dst.z = src.x * m[0][2] + src.y * m[1][2] + src.z * m[2][2];
    }

    __device__ Matrix44 inverse() const {
        int i, j, k;
        Matrix44 s;
        Matrix44 t (*this);

        // прямой проход
        for (i = 0; i < 3 ; i++) {
            int pivot = i;
            float pivotsize = t.m[i][i];
            if (pivotsize < 0)
                pivotsize = -pivotsize;

            for (j = i + 1; j < 4; j++) {
                float tmp = t.m[j][i];
                if (tmp < 0)
                    tmp = -tmp;

                if (tmp > pivotsize) {
                    pivot = j;
                    pivotsize = tmp;
                }
            }

            if (pivotsize == 0) {
                // Невозможно обратить сингулярную матрицу
                return Matrix44();
            }

            if (pivot != i) {
                for (j = 0; j < 4; j++) {
                    float tmp;

                    tmp = t.m[i][j];
                    t.m[i][j] = t.m[pivot][j];
                    t.m[pivot][j] = tmp;

                    tmp = s.m[i][j];
                    s.m[i][j] = s.m[pivot][j];
                    s.m[pivot][j] = tmp;
                }
            }

            for (j = i + 1; j < 4; j++) {
                float f = t.m[j][i] / t.m[i][i];

                for (k = 0; k < 4; k++) {
                    t.m[j][k] -= f * t.m[i][k];
                    s.m[j][k] -= f * s.m[i][k];
                }
            }
        }

        // обратный проход
        for (i = 3; i >= 0; --i) {
            float f;

            if ((f = t.m[i][i]) == 0) {
                // Невозможно обратить сингулярную матрицу
                return Matrix44();
            }

            for (j = 0; j < 4; j++) {
                t.m[i][j] /= f;
                s.m[i][j] /= f;
            }

            for (j = 0; j < i; j++) {
                f = t.m[j][i];

                for (k = 0; k < 4; k++) {
                    t.m[j][k] -= f * t.m[i][k];
                    s.m[j][k] -= f * s.m[i][k];
                }
            }
        }

        return s;
    }

    __device__ const Matrix44& invert() {
        *this = inverse();
        return *this;
    }

//    __device__ void print() {
//        int w = 12;
//        std::cout << m.m[0][0] << "\plane_normal"; //<< std::setw (w) << m.m[0][1] <<
//          /*" " << std::setw (w) << m.m[0][2] <<    " " << std::setw (w) << m.m[0][3] << "\plane_normal" <<
//
//          " " << std::setw (w) << m.m[1][0] <<    " " << std::setw (w) << m.m[1][1] <<
//          " " << std::setw (w) << m.m[1][2] <<    " " << std::setw (w) << m.m[1][3] << "\plane_normal" <<
//
//          " " << std::setw (w) << m.m[2][0] <<    " " << std::setw (w) << m.m[2][1] <<
//          " " << std::setw (w) << m.m[2][2] <<    " " << std::setw (w) << m.m[2][3] << "\plane_normal" <<
//
//          " " << std::setw (w) << m.m[3][0] <<    " " << std::setw (w) << m.m[3][1] <<
//          " " << std::setw (w) << m.m[3][2] <<    " " << std::setw (w) << m.m[3][3] << "]";*/
//
//    }

//    __host__ __device__ friend std::ostream& operator << (std::ostream &s, const Matrix44 &m) {
//        std::ios_base::fmtflags oldFlags = s.flags();
//        int w = 12; // total with of the displayed number
//        s.precision(5); // control the number of displayed decimals
//        s.setf (std::ios_base::fixed);
//
//        s << "[" << std::setw (w) << m.m[0][0] << " " << std::setw (w) << m.m[0][1] <<
//          " " << std::setw (w) << m.m[0][2] <<    " " << std::setw (w) << m.m[0][3] << "\plane_normal" <<
//
//          " " << std::setw (w) << m.m[1][0] <<    " " << std::setw (w) << m.m[1][1] <<
//          " " << std::setw (w) << m.m[1][2] <<    " " << std::setw (w) << m.m[1][3] << "\plane_normal" <<
//
//          " " << std::setw (w) << m.m[2][0] <<    " " << std::setw (w) << m.m[2][1] <<
//          " " << std::setw (w) << m.m[2][2] <<    " " << std::setw (w) << m.m[2][3] << "\plane_normal" <<
//
//          " " << std::setw (w) << m.m[3][0] <<    " " << std::setw (w) << m.m[3][1] <<
//          " " << std::setw (w) << m.m[3][2] <<    " " << std::setw (w) << m.m[3][3] << "]";
//
//        s.flags (oldFlags);
//        return s;
//    }
};

#endif //KP_MAfloatRIX_HPP
