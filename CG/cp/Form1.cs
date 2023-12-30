using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;
using System.Globalization;

namespace kp_work
{
    public partial class Form1 : Form
    {
        Graphics g;
        int N = 0;
        int oriN = 0;
        float zoomFactor = 2;
        public float height, width;

        List<List<Point>> contrPts;
        List<List<Point>> surfacePts;
        TransformMatrix matrix = new TransformMatrix();
        NURBS nurbs;

        float xStart;
        float yStart;
        float xEnd;
        float yEnd;

        Timer timerAnim = new Timer();

        public Form1()
        {
            InitializeComponent();
            timerAnim.Interval = 40;
            timerAnim.Tick += TimerAnim_Tick;
            ResizeRedraw = true;
            this.MouseWheel += new System.Windows.Forms.MouseEventHandler(this.form_MouseWheel);
        }

        //автоанимация
        private void TimerAnim_Tick(object sender, EventArgs e)
        {
            matrix.Mult(TransformMatrix.RotationY(6));
            this.Refresh();
        }

        //прокрутка колёсиком мыши
        private void form_MouseWheel(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            if (e.Delta > 0)
                zoomFactor *= 1.2f;
            else
                zoomFactor /= 1.2f;
            this.Refresh();
        }

        public void getPoints()
        {
            var filePath = string.Empty;
            filePath = @"C:\Users\Sergey\Desktop\Лихарев Сергей\МАИ\Информатика\CG\KP\kp_work\test1.txt";
            ParseFile(filePath);
            RecomputeSurface();
            this.Refresh();
        }
        
        //"распознавание" файла
        private void ParseFile(string filePath)
        {
            using (StreamReader reader = new StreamReader(filePath))
            {
                CultureInfo ci = CultureInfo.InvariantCulture;
                string[] parsed;

                int n, m;
                parsed = reader.ReadLine().Split();
                n = int.Parse(parsed[0]);
                m = int.Parse(parsed[1]);
                oriN = Math.Max(n, m);
                N = 2 * oriN;
                contrPts = new List<List<Point>>();
                for (int i = 0; i < n; i++)
                {
                    List<Point> row = new List<Point>();
                    parsed = reader.ReadLine().Trim().Split(" \t".ToCharArray(), StringSplitOptions.RemoveEmptyEntries);
                    for (int j = 0; j < m; j++)
                    {
                        row.Add(new Point(
                                float.Parse(parsed[j * 3], ci),
                                float.Parse(parsed[j * 3 + 1], ci),
                                float.Parse(parsed[j * 3 + 2], ci), 1
                        ));
                    }
                    contrPts.Add(row);
                }
                labelN.Text = N.ToString();
                nurbs = new NURBS(4, 4, contrPts);
            }
        }

        //перевычисление поверхностей при изменении её параметров
        private void RecomputeSurface()
        {
            surfacePts = nurbs.getSurfacePts(N);
            this.Refresh();
        }

        private void Form1_Shown(object sender, EventArgs e)
        {
            getPoints();
            labelN.Text = N.ToString();
        }
        
        private void Form1_Paint(object sender, PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.Clear(Color.Black);
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            this.height = this.Height - 100;
            this.width = this.Width;
           
            matrix.matrix[3, 3] = (int)(Math.Min(width, height) / zoomFactor);
            Pencil.DrawSurface(surfacePts, matrix, g, width, height);
        }

        //перетаскивание
        private void Form1_MouseDown(object sender, MouseEventArgs e)
        {
            xStart = xEnd = e.X;
            yStart = yEnd = e.Y;
        }

        //слайдер
        private void trackBar1_ValueChanged(object sender, EventArgs e)
        {
            N = 2*trackBarN.Value*oriN;
            labelN.Text = N.ToString();
            RecomputeSurface();
        }

        //загрузить
        private void загрузитьToolStripMenuItem_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.Filter = "txt files (*.txt)|*.txt|All files (*.*)|*.*";
                openFileDialog.FilterIndex = 2;
                openFileDialog.RestoreDirectory = true;

                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    string filePath = openFileDialog.FileName;
                    ParseFile(filePath);
                    RecomputeSurface();
                    this.Refresh();
                }
            }
        }

        //включить/отключить автоанимацию
        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox1.Checked)
                timerAnim.Start();
            else
                timerAnim.Stop();
        }

        //выход
        private void button1_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        //сохранить
        private void сохранитьToolStripMenuItem_Click(object sender, EventArgs e)
        {
            CultureInfo ci = CultureInfo.InvariantCulture;
            using (SaveFileDialog saveFileDialog = new SaveFileDialog())
            {
                saveFileDialog.Filter = "txt files (*.txt)|*.txt|All files (*.*)|*.*";
                saveFileDialog.FilterIndex = 2;
                saveFileDialog.RestoreDirectory = true;

                if (saveFileDialog.ShowDialog() == DialogResult.OK)
                {
                    string filePath = saveFileDialog.FileName;
                    TextWriter txt = new StreamWriter(filePath);
                    int n = contrPts.Count, m = contrPts[0].Count;
                    txt.WriteLine("{0} {1}", n, m);
                    for (int i = 0; i < n; i++)
                    {
                        for (int j = 0; j < m; j++)
                        {
                            txt.Write("{0} {1} {2} ", contrPts[i][j].x.ToString(ci), contrPts[i][j].y.ToString(ci), contrPts[i][j].z.ToString(ci));
                        }
                        txt.WriteLine();
                    }
                    txt.Close();
                }
            }
        }

        //сгенерировать
        private void сгенерироватьToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Random rnd = new Random((int)DateTime.Now.Ticks);
            int n = rnd.Next(10, 20);
            int m = n;
            oriN = Math.Max(n, m);
            N = 2 * oriN;
            contrPts = new List<List<Point>>();
            float dz = rnd.Next(1, 5);
            for (int i = 0; i < n; i++)
            {
                List<Point> row = new List<Point>();
                for (int j = 0; j < m; j++)
                {
                    row.Add(new Point((float)(i - n / 2), (float)j - m / 2, (float)rnd.NextDouble() * dz - dz / 2, 1));
                }
                contrPts.Add(row);
            }
            labelN.Text = N.ToString();
            nurbs = new NURBS(4, 4, contrPts);
            RecomputeSurface();
            this.Refresh();
        }

        //двигаем мышкой
        private void Form1_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                xEnd = e.X;
                yEnd = e.Y;

                float dX = xEnd - xStart;
                float dY = yEnd - yStart;

                matrix.Mult(TransformMatrix.RotationY(dX));
                matrix.Mult(TransformMatrix.RotationX(dY));

                xStart = xEnd;
                yStart = yEnd;

                this.Refresh();
            }
        }
    }
}