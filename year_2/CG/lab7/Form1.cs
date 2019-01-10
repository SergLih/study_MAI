using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace lab7
{
    public partial class Form1 : Form
    {
        List<PointF> pts = new List<PointF>();
        Graphics g;
        //настройка графика
        int pt_size = 4;
        Pen pen1 = new Pen(Brushes.Violet, 2);
        Pen pen2 = new Pen(Brushes.Red, 2);
        float tension = 0;

        //создание и загрузка формы
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            g = panelGr.CreateGraphics();
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
        }

        //при нажатии на область добавляется точка и перерисовка
        private void panelGr_MouseClick(object sender, MouseEventArgs e)
        {
            pts.Add(e.Location);
            updateGraphics();
        }

        //обновление графика
        private void updateGraphics()
        {
            g.Clear(Color.White);
            foreach (PointF pt in pts)
            {
                g.FillEllipse(Brushes.Blue, pt.X - pt_size, pt.Y - pt_size, pt_size * 2, pt_size * 2);
            }

            if (pts.Count >= 4)
            {
                if (radioButtonBuiltIn.Checked)
                {
                    g.DrawCurve(pen1, pts.ToArray(), tension);
                }
                else
                {
                    PointF[] splinePts = CardinalInterpolator.GetSpline(pts, tension);
                    g.DrawLines(pen2, splinePts);
                }
            }
        }
        
        //обновление параметра натяжения при движении слайдера
        private void trackBarTension_ValueChanged(object sender, EventArgs e)
        {
            tension = trackBarTension.Value / 10f;
            labelTension.Text = tension.ToString("F1");
            updateGraphics();
        }

        //кнопка "clear"
        private void buttonClear_Click(object sender, EventArgs e)
        {
            pts.Clear();
            g.Clear(Color.White);
        }

        //при перестановке выбора режима рисования
        private void radioButton2_CheckedChanged(object sender, EventArgs e)
        {
            updateGraphics();
        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            updateGraphics();
        }
    }
}
