using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace kp_work
{
    public class Point
    {
        public float x;
        public float y;
        public float z;
        public float w;


        public Point(float x, float y, float z, float w)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        public Point(Point p)
        {
            this.x = p.x;
            this.y = p.y;
        }

        public static Point Sum(Point v1, Point v2)
        {
            return new Point(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v2.w);
        }

        public static Point Sub(Point v1, Point v2)
        {
            return new Point(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z, v2.w);
        }

        public static Point Mul(Point v1, float m)
        {
            return new Point(v1.x * m, v1.y * m, v1.z * m, v1.w);
        }


        public Point Scale()
        {
            return new Point(x * w, y * w, z * w, 1);
        }
    }
}
