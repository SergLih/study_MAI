using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace kp_work
{
    public class Pencil
    {
        Pencil()
        {
        }

        public static void DrawList(List<Point> points, TransformMatrix matrix, System.Drawing.Graphics canvas, float width, float height)
        {
            List<Point> transformed = new List<Point>();
            for (int i = 0; i < points.Count; i++)
            {
                transformed.Add(matrix.Transform(points[i]).Scale());
            }

            float xCenter = width / 2;
            float yCenter = height / 2;

            for (int i = 0; i < transformed.Count - 1; i++)
            {
                Point p1 = transformed[i];
                Point p2 = transformed[i + 1];
                
                canvas.DrawLine(System.Drawing.Pens.Red, 
                    xCenter + p1.x, yCenter - p1.y, xCenter + p2.x, yCenter - p2.y);
            }
        }

        public static void DrawSurface(List<List<Point>> surface, TransformMatrix matrix, System.Drawing.Graphics canvas, float width, float height)
        {
            List<List<Point>> points = new List<List<Point>>();

            for (int i = 0; i < surface.Count; i++)
            {
                List<Point> tmp = new List<Point>();

                List<Point> current = surface[i];
                for (int j = 0; j < surface[i].Count; j++)
                {
                    tmp.Add(matrix.Transform(current[j]).Scale());
                }
                points.Add(tmp);
            }
            
            float xCenter = width / 2;
            float yCenter = height / 2;

            for (int i = 0; i < points.Count - 1; i++)
            {
                List<Point> cur = points[i];
                List<Point> next = points[i + 1];
                for (int j = 0; j < cur.Count - 1; j++)
                {
                    DrawRectangle(cur[j], cur[j + 1], next[j + 1], next[j], canvas, xCenter, yCenter, 
                        System.Drawing.Color.DodgerBlue);
                }
            }
        }

        private static void DrawRectangle(Point p1, Point p2, Point p3, Point p4, 
                                            System.Drawing.Graphics g, float xCenter, float yCenter, System.Drawing.Color color)
        {
            System.Drawing.Pen p = new System.Drawing.Pen(color, 1.5f);
            g.DrawLine(p, xCenter + p1.x, yCenter - p1.y, xCenter + p2.x, yCenter - p2.y);
            g.DrawLine(p, xCenter + p2.x, yCenter - p2.y, xCenter + p3.x, yCenter - p3.y);
            g.DrawLine(p, xCenter + p3.x, yCenter - p3.y, xCenter + p4.x, yCenter - p4.y);
            g.DrawLine(p, xCenter + p4.x, yCenter - p4.y, xCenter + p1.x, yCenter - p1.y);
        }

        private static void DrawTriangle(Point p1, Point p2, Point p3, System.Drawing.Graphics g, 
                                         float xCenter, float yCenter)
        {
            g.DrawLine(System.Drawing.Pens.Green, xCenter + p1.x, yCenter - p1.y, xCenter + p2.x, yCenter - p2.y);
            g.DrawLine(System.Drawing.Pens.Green, xCenter + p1.x, yCenter - p1.y, xCenter + p3.x, yCenter - p3.y);
            g.DrawLine(System.Drawing.Pens.Green, xCenter + p3.x, yCenter - p3.y, xCenter + p2.x, yCenter - p2.y);
        }
    }
}
