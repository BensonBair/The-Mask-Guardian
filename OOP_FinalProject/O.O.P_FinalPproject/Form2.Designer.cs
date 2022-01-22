namespace O.O.P_FinalPproject
{
    partial class Form2
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.resultPic = new System.Windows.Forms.PictureBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.lbl_hasMask = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.lbl_bodyTemp = new System.Windows.Forms.Label();
            this.lbl_employeeID = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.resultPic)).BeginInit();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // resultPic
            // 
            this.resultPic.Location = new System.Drawing.Point(150, 80);
            this.resultPic.Name = "resultPic";
            this.resultPic.Size = new System.Drawing.Size(500, 333);
            this.resultPic.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.resultPic.TabIndex = 0;
            this.resultPic.TabStop = false;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.lbl_hasMask);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.lbl_bodyTemp);
            this.groupBox1.Controls.Add(this.lbl_employeeID);
            this.groupBox1.Location = new System.Drawing.Point(250, 439);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(300, 182);
            this.groupBox1.TabIndex = 1;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Info";
            // 
            // lbl_hasMask
            // 
            this.lbl_hasMask.Font = new System.Drawing.Font("新細明體", 18F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.lbl_hasMask.Location = new System.Drawing.Point(64, 137);
            this.lbl_hasMask.Name = "lbl_hasMask";
            this.lbl_hasMask.Size = new System.Drawing.Size(181, 30);
            this.lbl_hasMask.TabIndex = 4;
            this.lbl_hasMask.Text = "hasMask";
            this.lbl_hasMask.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Times New Roman", 15F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.label3.Location = new System.Drawing.Point(132, 78);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(0, 25);
            this.label3.TabIndex = 3;
            // 
            // lbl_bodyTemp
            // 
            this.lbl_bodyTemp.Font = new System.Drawing.Font("新細明體", 18F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.lbl_bodyTemp.Location = new System.Drawing.Point(64, 87);
            this.lbl_bodyTemp.Name = "lbl_bodyTemp";
            this.lbl_bodyTemp.Size = new System.Drawing.Size(171, 30);
            this.lbl_bodyTemp.TabIndex = 1;
            this.lbl_bodyTemp.Text = "Body Temp. :";
            this.lbl_bodyTemp.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // lbl_employeeID
            // 
            this.lbl_employeeID.Font = new System.Drawing.Font("新細明體", 18F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.lbl_employeeID.ForeColor = System.Drawing.SystemColors.ControlText;
            this.lbl_employeeID.Location = new System.Drawing.Point(64, 37);
            this.lbl_employeeID.Name = "lbl_employeeID";
            this.lbl_employeeID.Size = new System.Drawing.Size(181, 30);
            this.lbl_employeeID.TabIndex = 0;
            this.lbl_employeeID.Text = "Employee ID: ";
            this.lbl_employeeID.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // Form2
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.ActiveCaption;
            this.ClientSize = new System.Drawing.Size(782, 653);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.resultPic);
            this.Name = "Form2";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Result Info";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form2_FormClosing);
            ((System.ComponentModel.ISupportInitialize)(this.resultPic)).EndInit();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox resultPic;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Label lbl_bodyTemp;
        private System.Windows.Forms.Label lbl_employeeID;
        private System.Windows.Forms.Label lbl_hasMask;
        private System.Windows.Forms.Label label3;
    }
}