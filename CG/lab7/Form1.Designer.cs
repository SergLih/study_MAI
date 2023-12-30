namespace lab7
{
    partial class Form1
    {
        /// <summary>
        /// Обязательная переменная конструктора.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Освободить все используемые ресурсы.
        /// </summary>
        /// <param name="disposing">истинно, если управляемый ресурс должен быть удален; иначе ложно.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Код, автоматически созданный конструктором форм Windows

        /// <summary>
        /// Требуемый метод для поддержки конструктора — не изменяйте 
        /// содержимое этого метода с помощью редактора кода.
        /// </summary>
        private void InitializeComponent()
        {
            this.panel1 = new System.Windows.Forms.Panel();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.radioButtonCustom = new System.Windows.Forms.RadioButton();
            this.radioButtonBuiltIn = new System.Windows.Forms.RadioButton();
            this.buttonClear = new System.Windows.Forms.Button();
            this.labelTension = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.trackBarTension = new System.Windows.Forms.TrackBar();
            this.panelGr = new System.Windows.Forms.Panel();
            this.panel1.SuspendLayout();
            this.groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarTension)).BeginInit();
            this.SuspendLayout();
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.groupBox1);
            this.panel1.Controls.Add(this.buttonClear);
            this.panel1.Controls.Add(this.labelTension);
            this.panel1.Controls.Add(this.label1);
            this.panel1.Controls.Add(this.trackBarTension);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Top;
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(478, 62);
            this.panel1.TabIndex = 0;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.radioButtonCustom);
            this.groupBox1.Controls.Add(this.radioButtonBuiltIn);
            this.groupBox1.Location = new System.Drawing.Point(215, 3);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(150, 46);
            this.groupBox1.TabIndex = 4;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Drawing Mode";
            // 
            // radioButtonCustom
            // 
            this.radioButtonCustom.AutoSize = true;
            this.radioButtonCustom.Location = new System.Drawing.Point(79, 20);
            this.radioButtonCustom.Name = "radioButtonCustom";
            this.radioButtonCustom.Size = new System.Drawing.Size(59, 17);
            this.radioButtonCustom.TabIndex = 0;
            this.radioButtonCustom.Text = "custom";
            this.radioButtonCustom.UseVisualStyleBackColor = true;
            this.radioButtonCustom.CheckedChanged += new System.EventHandler(this.radioButton2_CheckedChanged);
            // 
            // radioButtonBuiltIn
            // 
            this.radioButtonBuiltIn.AutoSize = true;
            this.radioButtonBuiltIn.Checked = true;
            this.radioButtonBuiltIn.Location = new System.Drawing.Point(7, 20);
            this.radioButtonBuiltIn.Name = "radioButtonBuiltIn";
            this.radioButtonBuiltIn.Size = new System.Drawing.Size(55, 17);
            this.radioButtonBuiltIn.TabIndex = 0;
            this.radioButtonBuiltIn.TabStop = true;
            this.radioButtonBuiltIn.Text = "built-in";
            this.radioButtonBuiltIn.UseVisualStyleBackColor = true;
            this.radioButtonBuiltIn.CheckedChanged += new System.EventHandler(this.radioButton1_CheckedChanged);
            // 
            // buttonClear
            // 
            this.buttonClear.Location = new System.Drawing.Point(382, 17);
            this.buttonClear.Name = "buttonClear";
            this.buttonClear.Size = new System.Drawing.Size(75, 23);
            this.buttonClear.TabIndex = 3;
            this.buttonClear.Text = "Clear";
            this.buttonClear.UseVisualStyleBackColor = true;
            this.buttonClear.Click += new System.EventHandler(this.buttonClear_Click);
            // 
            // labelTension
            // 
            this.labelTension.AutoSize = true;
            this.labelTension.Location = new System.Drawing.Point(66, 19);
            this.labelTension.Name = "labelTension";
            this.labelTension.Size = new System.Drawing.Size(22, 13);
            this.labelTension.TabIndex = 2;
            this.labelTension.Text = "0,0";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 19);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(48, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Tension:";
            // 
            // trackBarTension
            // 
            this.trackBarTension.LargeChange = 10;
            this.trackBarTension.Location = new System.Drawing.Point(94, 17);
            this.trackBarTension.Maximum = 30;
            this.trackBarTension.Name = "trackBarTension";
            this.trackBarTension.Size = new System.Drawing.Size(115, 45);
            this.trackBarTension.TabIndex = 0;
            this.trackBarTension.TickFrequency = 5;
            this.trackBarTension.ValueChanged += new System.EventHandler(this.trackBarTension_ValueChanged);
            // 
            // panelGr
            // 
            this.panelGr.BackColor = System.Drawing.Color.White;
            this.panelGr.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panelGr.Location = new System.Drawing.Point(0, 62);
            this.panelGr.Name = "panelGr";
            this.panelGr.Size = new System.Drawing.Size(478, 287);
            this.panelGr.TabIndex = 1;
            this.panelGr.MouseClick += new System.Windows.Forms.MouseEventHandler(this.panelGr_MouseClick);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(478, 349);
            this.Controls.Add(this.panelGr);
            this.Controls.Add(this.panel1);
            this.Name = "Form1";
            this.Text = "Lab7";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarTension)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Label labelTension;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TrackBar trackBarTension;
        private System.Windows.Forms.Panel panelGr;
        private System.Windows.Forms.Button buttonClear;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.RadioButton radioButtonCustom;
        private System.Windows.Forms.RadioButton radioButtonBuiltIn;
    }
}

