// Фамилия: Лихарев С.С.
// Группа:  M80-307Б
// Программа, реализующая NURBS поверхность порядка 4x4


using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace kp_work
{
    public class NURBS
    {
        private int k;  //порядок B-сплайна по направлению u
        private int q;  //порядок B-сплайна по направлению v
        private int m;  //кол-во контрольных точек по направлению u
        private int n;  //кол-во контрольных точек по направлению v
        private List<List<Point>> contrPts;      //Массив контрольных точек
        private int[] uPts;      //Массив узловых точек по направлению u
        private int[] vPts;      //Массив узловых точек по направлению v

        /// <summary>
        /// Конструктор поверхности NURBS
        /// </summary>
        /// <param name="k">порядок B-сплайна по направлению u</param>
        /// <param name="q">порядок B-сплайна по направлению v</param>
        /// <param name="p">массив контрольных точек (каждая -- три координаты и вес)</param>
        public NURBS(int k, int q, List<List<Point>> p)
        {
            this.k = k;
            this.q = q;
            contrPts = p;
            m = p.Count;
            n = p[0].Count;
            uPts = new int[m + k+1];
            vPts = new int[n + q+1];
            
            for (int j = 0; j < m + k + 1; j++)
                if (j < k)
                    uPts[j] = 0;
                else if (j >= m)
                    uPts[j] = m - k + 1;
                else
                    uPts[j] = j - k + 1;
            
            for (int i = 0; i < n + q + 1; i++)
                if (i < q)
                    vPts[i] = 0;
                else if (i >= n)
                    vPts[i] = n - q + 1;
                else
                    vPts[i] = i - q + 1;
        }

        //формула Кокса – де Бура
        private float CoxDeBoor(float p, int i, int k, int[] pts)
        {
            if (k == 1)
                return (pts[i] - 0.001f <= p && p <= pts[i + 1] + 0.001f) ? 1 : 0;

            float div1 = pts[i + k - 1] - pts[i];
            float div2 = pts[i + k] - pts[i + 1];
            float res1 = (div1 > 0) ? (p - pts[i]) / div1 * CoxDeBoor(p, i, k - 1, pts) : 0;
            float res2 = (div2 > 0) ? (pts[i + k] - p) / div2 * CoxDeBoor(p, i + 1, k - 1, pts) : 0;
            return res1 + res2;
        }

        //вычисление радиус-вектора
        private Point r(float u, float v)
        {
            Point res = new Point(0, 0, 0, 1);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++)
                    res = Point.Sum(res, Point.Mul(contrPts[i][j], CoxDeBoor(v, i, q, vPts) * CoxDeBoor(u, j, k, uPts)));
            return res;
        }

        //генерация координат поверхности в заданном числе узловых точек
        public List<List<Point>> getSurfacePts(int k_grid)
        {
            float u_step = (m - k + 1) / (float)(k_grid - 1);
            float v_step = (n - q + 1) / (float)(k_grid - 1);
            List<List<Point>> res = new List<List<Point>>();
            float u, v; int i, j;
            for (i = 0, u = 0; i < k_grid; i++, u += u_step)
            {
                List<Point> row = new List<Point>();
                for (j = 0, v = 0; j < k_grid; j++, v += v_step)
                    row.Add(r(u, v));
                res.Add(row);
            }
            return res;
        }
    }
}
