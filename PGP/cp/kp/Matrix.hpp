#ifndef KP_MATRIX_HPP
#define KP_MATRIX_HPP

#include <cstdint>
#include <iomanip>
#include "vector.hpp"

template<typename T>
class Matrix44 {
public:

    T x[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    static const Matrix44 kIdentity;

    Matrix44() {}

    Matrix44(T data[4][4]) {
        for (uint8_t i = 0; i < 4; ++i) {
            for (uint8_t j = 0; j < 4; ++j) {
                x[i][j] = data[i][j];
            }
        }
    }

    const T* operator [] (uint8_t i) const { return x[i]; }
    T* operator [] (uint8_t i) { return x[i]; }

    // Умножить текущую матрицу на другую (справа)
    Matrix44 operator * (const Matrix44& v) const {
        Matrix44 tmp;
        multiply (*this, v, tmp);

        return tmp;
    }

    static void multiply(const Matrix44<T> &a, const Matrix44& b, Matrix44 &c) {
        for (uint8_t i = 0; i < 4; ++i) {
            for (uint8_t j = 0; j < 4; ++j) {
                c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] +
                          a[i][2] * b[2][j] + a[i][3] * b[3][j];
            }
        }
    }

    Matrix44 transposed() const {
        Matrix44 t;
        for (uint8_t i = 0; i < 4; ++i) {
            for (uint8_t j = 0; j < 4; ++j) {
                t[i][j] = x[j][i];
            }
        }
        return t;
    }

    template<typename S>
    void multVecMatrix(const vec<3, S> &src, vec<3, S> &dst) const {
        S a, b, c, w;
        a = src.x * x[0][0] + src.y * x[1][0] + src.z * x[2][0] + x[3][0];
        b = src.x * x[0][1] + src.y * x[1][1] + src.z * x[2][1] + x[3][1];
        c = src.x * x[0][2] + src.y * x[1][2] + src.z * x[2][2] + x[3][2];
        w = src.x * x[0][3] + src.y * x[1][3] + src.z * x[2][3] + x[3][3];

        dst.x = a / w;
        dst.y = b / w;
        dst.z = c / w;
    }

    //не учитываем коэф-ты отвечающие за сдвиг (translation)
    template<typename S>
    void multDirMatrix(const vec<3, S> &src, vec<3, S> &dst) const {
        dst.x = src.x * x[0][0] + src.y * x[1][0] + src.z * x[2][0];
        dst.y = src.x * x[0][1] + src.y * x[1][1] + src.z * x[2][1];
        dst.z = src.x * x[0][2] + src.y * x[1][2] + src.z * x[2][2];
    }

    Matrix44 inverse() const {
        int i, j, k;
        Matrix44 s;
        Matrix44 t (*this);

        // прямой проход
        for (i = 0; i < 3 ; i++) {
            int pivot = i;
            T pivotsize = t[i][i];
            if (pivotsize < 0)
                pivotsize = -pivotsize;

            for (j = i + 1; j < 4; j++) {
                T tmp = t[j][i];
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
                    T tmp;

                    tmp = t[i][j];
                    t[i][j] = t[pivot][j];
                    t[pivot][j] = tmp;

                    tmp = s[i][j];
                    s[i][j] = s[pivot][j];
                    s[pivot][j] = tmp;
                }
            }

            for (j = i + 1; j < 4; j++) {
                T f = t[j][i] / t[i][i];

                for (k = 0; k < 4; k++) {
                    t[j][k] -= f * t[i][k];
                    s[j][k] -= f * s[i][k];
                }
            }
        }

        // обратный проход
        for (i = 3; i >= 0; --i) {
            T f;

            if ((f = t[i][i]) == 0) {
                // Невозможно обратить сингулярную матрицу
                return Matrix44();
            }

            for (j = 0; j < 4; j++) {
                t[i][j] /= f;
                s[i][j] /= f;
            }

            for (j = 0; j < i; j++) {
                f = t[j][i];

                for (k = 0; k < 4; k++) {
                    t[j][k] -= f * t[i][k];
                    s[j][k] -= f * s[i][k];
                }
            }
        }

        return s;
    }

    const Matrix44<T>& invert() {
        *this = inverse();
        return *this;
    }

    friend std::ostream& operator << (std::ostream &s, const Matrix44 &m) {
        std::ios_base::fmtflags oldFlags = s.flags();
        int w = 12; // total with of the displayed number
        s.precision(5); // control the number of displayed decimals
        s.setf (std::ios_base::fixed);

        s << "[" << std::setw (w) << m[0][0] << " " << std::setw (w) << m[0][1] <<
          " " << std::setw (w) << m[0][2] <<    " " << std::setw (w) << m[0][3] << "\n" <<

          " " << std::setw (w) << m[1][0] <<    " " << std::setw (w) << m[1][1] <<
          " " << std::setw (w) << m[1][2] <<    " " << std::setw (w) << m[1][3] << "\n" <<

          " " << std::setw (w) << m[2][0] <<    " " << std::setw (w) << m[2][1] <<
          " " << std::setw (w) << m[2][2] <<    " " << std::setw (w) << m[2][3] << "\n" <<

          " " << std::setw (w) << m[3][0] <<    " " << std::setw (w) << m[3][1] <<
          " " << std::setw (w) << m[3][2] <<    " " << std::setw (w) << m[3][3] << "]";

        s.flags(oldFlags);
        return s;
    }
};

typedef Matrix44<float> Matrix44f;


#endif //KP_MATRIX_HPP