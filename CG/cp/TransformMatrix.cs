using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace kp_work
{
    public class TransformMatrix
    {

        public double[,] matrix;

        public TransformMatrix(double[,] matrix)
        {
            this.matrix = matrix;
        }

        public TransformMatrix()
        {
            this.matrix = new double[4,4]{
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}
             };
        }

        public Point Transform(Point vector)
        {
            // All matrix elements from 00 to 33
            double _00 = this.matrix[0,0];
            double _01 = this.matrix[0,1];
            double _02 = this.matrix[0,2];
            double _03 = this.matrix[0,3];
            double _10 = this.matrix[1,0];
            double _11 = this.matrix[1,1];
            double _12 = this.matrix[1,2];
            double _13 = this.matrix[1,3];
            double _20 = this.matrix[2,0];
            double _21 = this.matrix[2,1];
            double _22 = this.matrix[2,2];
            double _23 = this.matrix[2,3];
            double _30 = this.matrix[3,0];
            double _31 = this.matrix[3,1];
            double _32 = this.matrix[3,2];
            double _33 = this.matrix[3,3];

            float x = (float) (vector.x * _00 + vector.y * _10 + vector.z * _20 + vector.w * _30);
            float y = (float) (vector.x * _01 + vector.y * _11 + vector.z * _21 + vector.w * _31);
            float z = (float) (vector.x * _02 + vector.y * _12 + vector.z * _22 + vector.w * _32);
            float h = (float) (vector.x * _03 + vector.y * _13 + vector.z * _23 + vector.w * _33);

            return new Point(x, y, z, h);
        }

        public void Mult(TransformMatrix m)
        {
            double[,] t = {
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}
        };
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        t[i,j] += this.matrix[i,k] * m.GetMatrix()[j,k];
                    }
                }
            }
            this.matrix = t;
        }

        public static TransformMatrix RotationY(double angle)
        {
            double rad = (angle % 360) * Math.PI / 180;
            double COS = Math.Cos(rad);
            double SIN = Math.Sin(rad);
            double[,] t = new double[,]{
                {COS,  0, -SIN, 0},
                {0,    1,    0, 0},
                {SIN,  0,  COS, 0},
                {0,    0,    0, 1}};
            return new TransformMatrix(t);
        }

        public static TransformMatrix RotationX(double angle)
        {
            double rad = (angle % 360) * Math.PI / 180;
            double COS = Math.Cos(rad);
            double SIN = Math.Sin(rad);
            double[,] t = new double[,]{
                {1,    0,    0, 0},
                {0,  COS,  SIN, 0},
                {0, -SIN,  COS, 0},
                {0,    0,    0, 1}};
            return new TransformMatrix(t);
        }

        public static TransformMatrix Screen(double width, double height)
        {
            double A = (width - 1.0) / 2.0;
            double B = (height - 1.0) / 2.0;
            double[,] t = new double[,]{
                {A, 0, 0, A},
                {0, -B, 0, B},
                {0, 0, 1, 0},
                {0, 0, 0, 1}};
            return new TransformMatrix(t);
        }

        public static TransformMatrix View(double x, double y, double z)
        {
            double[,] t = new double[,]{
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {x, y, z, 1}};
            return new TransformMatrix(t);
        }


        public double[,] GetMatrix()
        {
            return matrix;
        }
    }
}
