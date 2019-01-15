// Фамилия: Лихарев С.С.
// Группа:  M80-307Б
// Программа, реализующая Cardinal Spline 3-й степени из двух сегментов

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace lab7
{
    public class CardinalInterpolator
    {
        //многочлены Эрмита
        private static float h1(float t) { return 2 * t * t * t - 3 * t * t + 1; }
        private static float h2(float t) { return -2 * t * t * t + 3 * t * t; }
        private static float h3(float t) { return t * t * t - 2 * t * t + t; }
        private static float h4(float t) { return t * t * t - t * t; }

        //вычисляем один сегмент кривой между четырьмя точками
        private static PointF[] GetSplinePart(PointF P1, PointF P2, PointF P3, PointF P4, float tension = 0.5f, int nPoints = 100)
        {
            float epsylon = 1f / nPoints, x, y;
            List<PointF> points = new List<PointF>();
            for (float t = 0; t < 1; t += epsylon)
            {
                x = tension * (h3(t) * (P3.X - P1.X) + h4(t) * (P4.X - P2.X)) + h1(t) * P2.X + h2(t) * P3.X;
                y = tension * (h3(t) * (P3.Y - P1.Y) + h4(t) * (P4.Y - P2.Y)) + h1(t) * P2.Y + h2(t) * P3.Y;
                points.Add(new PointF(x, y));
            }
            return points.ToArray();
        }

        //вычисляем все сегменты, кроме двух крайних из набора точек
        public static PointF[] GetSpline(List<PointF> pts, float tension = 0.5f)
        {
            List<PointF> splinePts = new List<PointF>();
            for (int i = 1; i < pts.Count-2; i++)
            {
                splinePts.AddRange(GetSplinePart(pts[i - 1], pts[i], pts[i + 1], pts[i + 2], tension));
            }
            return splinePts.ToArray();
        }

    }
}
