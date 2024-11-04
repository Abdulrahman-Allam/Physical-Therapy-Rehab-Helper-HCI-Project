/*
	TUIO C# Demo - part of the reacTIVision project
	Copyright (c) 2005-2016 Martin Kaltenbrunner <martin@tuio.org>

	This program is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation; either version 2 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program; if not, write to the Free Software
	Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

//mine

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Net.Sockets;
using System.Text;
using System.Windows.Forms;
using TUIO;


public class Patient
{
    public string name = "";
    public string age = "";
    public string score = "";
    public string mac = "";

    public Patient(string name, string age, string score, string mac)
    {
        this.name = name;
        this.age = age;
        this.score = score;
        this.mac = mac;
    }

    public Patient()
    {

    }
}



public class TuioDemo : Form, TuioListener
{
    private TuioClient client;
    private Dictionary<long, TuioObject> objectList;
    private Dictionary<long, TuioCursor> cursorList;
    private Dictionary<long, TuioBlob> blobList;

    public static int width, height;
    private int screen_width = Screen.PrimaryScreen.Bounds.Width;
    private int screen_height = Screen.PrimaryScreen.Bounds.Height;
    private int w_top = 0;
    private int w_width = 640;
    private int w_height = 480;
    private int w_left = 0;
    public int prev_id = -1;
    private bool fullscreen;
    private bool verbose;

    public string serverIP = "localhost"; // IP address of the Python server
    public int port = 8000;               // Port number matching the Python server
    int flag = 0;
    Font font = new Font("Arial", 15.0f);
    SolidBrush fntBrush = new SolidBrush(Color.White);
    SolidBrush bgrBrush = new SolidBrush(Color.Black);
    SolidBrush curBrush = new SolidBrush(Color.Yellow);
    SolidBrush objBrush = new SolidBrush(Color.Purple);
    SolidBrush blbBrush = new SolidBrush(Color.Red);
    Pen curPen = new Pen(new SolidBrush(Color.Blue), 1);
    private string objectImagePath;
    private string backgroundImagePath;
    TcpClient client1;
    NetworkStream stream;
    string title = "PTRH-HCI";
    string prevP = "";

    bool patient = false;
    bool doctor = false;
    Patient currPatient;
    List<Patient> patients = new List<Patient>
    {
        new Patient("Hamza","19","600","A0:D0:5B:27:31:17"),
        new Patient("DHOM", "21", "555", "A0:D0:5B:27:31:12"),
        new Patient("Abood", "20", "500", "A0:D0:5B:27:31:13"),
        new Patient("Abdo", "21", "400", "A0:D0:5B:27:31:14"),
        new Patient("Assem", "24", "300", "A0:D0:5B:27:31:15"),
        new Patient("Seif", "21", "200", "A0:D0:5B:27:31:16"),
    };
    Patient logPatient = new Patient();





    private string selectedPatient;

    public TuioDemo(int port)
    {
        verbose = false;
        fullscreen = false;
        width = w_width;
        height = w_height;

        this.ClientSize = new System.Drawing.Size(width, height);
        this.Name = "TuioDemo";
        this.Text = title;

        this.Closing += new CancelEventHandler(Form_Closing);
        this.KeyDown += new KeyEventHandler(Form_KeyDown);

        this.SetStyle(ControlStyles.AllPaintingInWmPaint |
                        ControlStyles.UserPaint |
                        ControlStyles.DoubleBuffer, true);

        objectList = new Dictionary<long, TuioObject>(128);
        cursorList = new Dictionary<long, TuioCursor>(128);
        blobList = new Dictionary<long, TuioBlob>(128);

        client = new TuioClient(port);
        client.addTuioListener(this);

        client.connect();

        // Create a TCP/IP socket
        client1 = new TcpClient(serverIP, 8000);
        // Get the stream to send data
        stream = client1.GetStream();

    }

    private void Form_KeyDown(object sender, System.Windows.Forms.KeyEventArgs e)
    {

        if (e.KeyData == Keys.F1)
        {
            if (fullscreen == false)
            {

                width = screen_width;
                height = screen_height;

                w_left = this.Left;
                w_top = this.Top;

                this.FormBorderStyle = FormBorderStyle.None;
                this.Left = 0;
                this.Top = 0;
                this.Width = screen_width;
                this.Height = screen_height;

                fullscreen = true;
            }
            else
            {

                width = w_width;
                height = w_height;

                this.FormBorderStyle = FormBorderStyle.Sizable;
                this.Left = w_left;
                this.Top = w_top;
                this.Width = w_width;
                this.Height = w_height;

                fullscreen = false;
            }
        }
        else if (e.KeyData == Keys.Escape)
        {
            // Close everything
            stream.Close();
            client1.Close();
            this.Close();

        }
        else if (e.KeyData == Keys.V)
        {
            verbose = !verbose;
        }

        if (e.KeyData == Keys.P)
        {
            receiveSocket();
        }

    }

    private void Form_Closing(object sender, System.ComponentModel.CancelEventArgs e)
    {
        client.removeTuioListener(this);

        client.disconnect();
        System.Environment.Exit(0);
    }

    public void addTuioObject(TuioObject o)
    {
        lock (objectList)
        {
            objectList.Add(o.SessionID, o);
        }
        if (verbose) Console.WriteLine("add obj " + o.SymbolID + " (" + o.SessionID + ") " + o.X + " " + o.Y + " " + o.Angle);
    }

    public void updateTuioObject(TuioObject o)
    {

        if (verbose) Console.WriteLine("set obj " + o.SymbolID + " " + o.SessionID + " " + o.X + " " + o.Y + " " + o.Angle + " " + o.MotionSpeed + " " + o.RotationSpeed + " " + o.MotionAccel + " " + o.RotationAccel);
    }

    public void removeTuioObject(TuioObject o)
    {
        lock (objectList)
        {
            objectList.Remove(o.SessionID);
        }
        if (verbose) Console.WriteLine("del obj " + o.SymbolID + " (" + o.SessionID + ")");
    }

    public void addTuioCursor(TuioCursor c)
    {
        lock (cursorList)
        {
            cursorList.Add(c.SessionID, c);
        }
        if (verbose) Console.WriteLine("add cur " + c.CursorID + " (" + c.SessionID + ") " + c.X + " " + c.Y);
    }

    public void updateTuioCursor(TuioCursor c)
    {
        if (verbose) Console.WriteLine("set cur " + c.CursorID + " (" + c.SessionID + ") " + c.X + " " + c.Y + " " + c.MotionSpeed + " " + c.MotionAccel);
    }

    public void removeTuioCursor(TuioCursor c)
    {
        lock (cursorList)
        {
            cursorList.Remove(c.SessionID);
        }
        if (verbose) Console.WriteLine("del cur " + c.CursorID + " (" + c.SessionID + ")");
    }

    public void addTuioBlob(TuioBlob b)
    {
        lock (blobList)
        {
            blobList.Add(b.SessionID, b);
        }
        if (verbose) Console.WriteLine("add blb " + b.BlobID + " (" + b.SessionID + ") " + b.X + " " + b.Y + " " + b.Angle + " " + b.Width + " " + b.Height + " " + b.Area);
    }

    public void updateTuioBlob(TuioBlob b)
    {

        if (verbose) Console.WriteLine("set blb " + b.BlobID + " (" + b.SessionID + ") " + b.X + " " + b.Y + " " + b.Angle + " " + b.Width + " " + b.Height + " " + b.Area + " " + b.MotionSpeed + " " + b.RotationSpeed + " " + b.MotionAccel + " " + b.RotationAccel);
    }

    public void removeTuioBlob(TuioBlob b)
    {
        lock (blobList)
        {
            blobList.Remove(b.SessionID);
        }
        if (verbose) Console.WriteLine("del blb " + b.BlobID + " (" + b.SessionID + ")");
    }

    public void refresh(TuioTime frameTime)
    {
        receiveSocket();
        Invalidate();
    }

    protected override void OnPaintBackground(PaintEventArgs pevent)
    {
        // Getting the graphics object
        Graphics g = pevent.Graphics;


        // Draw the background
        g.FillRectangle(bgrBrush, new Rectangle(0, 0, width, height));

        if (patient)
        {
            g.FillRectangle(new SolidBrush(Color.Red), new Rectangle(230, 70, logPatient.name.Length * 40, 30));
            g.DrawString("welcome " + logPatient.name, font, Brushes.White,
                         new Point(230, 70));
            g.DrawString("your score: " + logPatient.score, font, Brushes.White,
                         new Point(230, 100));
        }
        else if (doctor)
        {
            int menuRadius = height / 4;  // Adjust size based on your design
            Point center = new Point(width / 2, height / 2);

            // Define patient names based on marker ID
            string[] patientNamesMarker1 = { patients[0].name, patients[1].name, patients[2].name, patients[3].name };
            string[] patientNamesMarker2 = { patients[4].name, patients[5].name, "Kareem Karam", "Tamer Hosny" };
            string[] patientNamesMarker3 = { "Abdulrahman Allam", "Hamza Moustafa", "Khalid Hassan", "Ahmed Saliba" };

            if (flag == 0)
            {
                // Draw the background
                g.FillRectangle(bgrBrush, new Rectangle(0, 0, width, height));

                // Draw the cursor path
                if (cursorList.Count > 0)
                {
                    lock (cursorList)
                    {
                        foreach (TuioCursor tcur in cursorList.Values)
                        {
                            List<TuioPoint> path = tcur.Path;
                            TuioPoint current_point = path[0];

                            for (int i = 0; i < path.Count; i++)
                            {
                                TuioPoint next_point = path[i];
                                g.DrawLine(curPen, current_point.getScreenX(width), current_point.getScreenY(height), next_point.getScreenX(width), next_point.getScreenY(height));
                                current_point = next_point;
                            }

                            g.FillEllipse(curBrush, current_point.getScreenX(width) - height / 100, current_point.getScreenY(height) - height / 100, height / 50, height / 50);
                            g.DrawString(tcur.CursorID + "", font, fntBrush, new PointF(tcur.getScreenX(width) - 10, tcur.getScreenY(height) - 10));
                        }
                    }
                }

                // Draw the circular menu

                g.DrawEllipse(Pens.Red, center.X - menuRadius, center.Y - menuRadius, menuRadius * 2, menuRadius * 2);

                // Draw arcs
                for (int i = 0; i < 4; i++)
                {
                    double angleStart = i * 90;
                    g.FillPie(Brushes.White, center.X - menuRadius, center.Y - menuRadius, menuRadius * 2, menuRadius * 2, (float)angleStart, 90);
                }

                if (objectList.Count > 0)
                {
                    lock (objectList)
                    {
                        foreach (TuioObject tobj in objectList.Values)
                        {
                            // Get the marker position
                            int markerX = tobj.getScreenX(width);
                            int markerY = tobj.getScreenY(height);

                            // Calculate the distance from the center of the circle to the marker
                            double distanceFromCenter = Math.Sqrt(Math.Pow(markerX - center.X, 2) + Math.Pow(markerY - center.Y, 2));



                            // Calculate the angle of the marker
                            double angle = Math.Atan2(markerY - center.Y, markerX - center.X) * (180.0 / Math.PI);
                            if (angle < 0) angle += 360; // Convert angle to 0-360 range

                            // Determine which arc the marker is in
                            int selectedQuadrant = (int)(angle / 90) % 4;

                            // Highlight the selected quadrant
                            g.FillPie(Brushes.GreenYellow, center.X - menuRadius, center.Y - menuRadius, menuRadius * 2, menuRadius * 2, selectedQuadrant * 90, 90);

                            // Display patient names based on the marker ID
                            string[] currentPatientNames = tobj.SymbolID == 1 ? patientNamesMarker1 :
                                                            tobj.SymbolID == 2 ? patientNamesMarker2 :
                                                            tobj.SymbolID == 3 ? patientNamesMarker3 : null;




                            if (currentPatientNames != null)
                            {
                                selectedPatient = currentPatientNames[selectedQuadrant];

                                g.FillRectangle(new SolidBrush(Color.DeepSkyBlue), new Rectangle(0, 5, selectedPatient.Length * 11, 30));
                                g.DrawString(currentPatientNames[selectedQuadrant], font, Brushes.White,
                                             new Point(0, 8));




                                if ((tobj.Angle * (180.0 / Math.PI) >= 77 && tobj.Angle * (180.0 / Math.PI) < 280))
                                {
                                    if (prevP != selectedPatient)
                                    {

                                        prevP = selectedPatient;
                                        for (int i = 0; i < patients.Count; i++)
                                        {
                                            if (patients[i].name == prevP)
                                            {
                                                currPatient = patients[i];
                                                break;
                                            }
                                        }
                                        flag = 1;
                                        sendSocket(tobj);

                                    }

                                }

                            }




                            // Existing object rendering logic
                            int ox = tobj.getScreenX(width);
                            int oy = tobj.getScreenY(height);
                            int size = height / 10;

                            g.TranslateTransform(ox, oy);
                            g.RotateTransform((float)(tobj.Angle / Math.PI * 180.0f));
                            g.TranslateTransform(-ox, -oy);

                            g.FillRectangle(objBrush, new Rectangle(ox - size / 2, oy - size / 2, size, size));

                            g.TranslateTransform(ox, oy);
                            g.RotateTransform(-1 * (float)(tobj.Angle / Math.PI * 180.0f));
                            g.TranslateTransform(-ox, -oy);

                            g.DrawString(tobj.SymbolID + "", font, fntBrush, new PointF(ox - 10, oy - 10));

                            // Existing object image drawing logic...
                        }
                    }
                }
            }
            else if (flag == 1)
            {
                this.StartPosition = FormStartPosition.CenterScreen;
                this.BackColor = Color.Black;
                Rectangle backBoxRect = new Rectangle(10, 10, 150, this.Height - 60);
                using (Brush backBoxBrush = new SolidBrush(Color.Green))
                {
                    g.FillRectangle(backBoxBrush, backBoxRect);
                }

                // Draw the back label inside the back box
                using (Font backFont = new Font("Arial", 24, FontStyle.Bold))
                using (Brush backTextBrush = new SolidBrush(Color.White))
                {
                    StringFormat sf = new StringFormat
                    {
                        Alignment = StringAlignment.Center,
                        LineAlignment = StringAlignment.Center
                    };
                    g.DrawString("<- Back", backFont, backTextBrush, backBoxRect, sf);
                }

                // Draw the ID label
                using (Font labelFont = new Font("Arial", 24, FontStyle.Bold))
                using (Brush labelBrush = new SolidBrush(Color.White))
                {
                    // Center X-coordinate calculation for labels
                    int labelX = (this.ClientSize.Width - 100) / 2;

                    // ID Label
                    g.DrawString("ID: " + currPatient.mac, labelFont, labelBrush, labelX, 100);

                    // Name Label
                    g.DrawString("Name: " + currPatient.name, labelFont, labelBrush, labelX, 200);

                    // Score Label
                    g.DrawString("Score: " + currPatient.score, labelFont, labelBrush, labelX, 300);
                }


                // Draw TUIO Marker cursors last to bring them "above" other controls
                if (objectList.Count > 0)
                {
                    lock (objectList)
                    {
                        foreach (TuioObject tobj in objectList.Values)
                        {
                            // Get marker position and draw markers
                            int markerX = tobj.getScreenX(width);
                            int markerY = tobj.getScreenY(height);

                            // Calculate angle and position
                            double distanceFromCenter = Math.Sqrt(Math.Pow(markerX - center.X, 2) + Math.Pow(markerY - center.Y, 2));
                            double angle = Math.Atan2(markerY - center.Y, markerX - center.X) * (180.0 / Math.PI);
                            if (angle < 0) angle += 360;

                            // Draw TUIO Marker on top
                            int ox = tobj.getScreenX(width);
                            int oy = tobj.getScreenY(height);
                            int size = height / 10;




                            if (backBoxRect.Contains(markerX, markerY))
                            {
                                // Check if the rotation angle is close to 90 degrees
                                if ((tobj.Angle * (180.0 / Math.PI) >= 77 && tobj.Angle * (180.0 / Math.PI) < 280)) // Adjust threshold as needed
                                {
                                    flag = 0;
                                }
                            }






                            g.TranslateTransform(ox, oy);
                            g.RotateTransform((float)(tobj.Angle / Math.PI * 180.0f));
                            g.TranslateTransform(-ox, -oy);

                            g.FillRectangle(objBrush, new Rectangle(ox - size / 2, oy - size / 2, size, size));

                            g.TranslateTransform(ox, oy);
                            g.RotateTransform(-1 * (float)(tobj.Angle / Math.PI * 180.0f));
                            g.TranslateTransform(-ox, -oy);

                            g.DrawString(tobj.SymbolID + "", font, fntBrush, new PointF(ox - 10, oy - 10));
                        }
                    }
                }



            }





            // Draw the blobs
            if (blobList.Count > 0)
            {
                lock (blobList)
                {
                    foreach (TuioBlob tblb in blobList.Values)
                    {
                        int bx = tblb.getScreenX(width);
                        int by = tblb.getScreenY(height);
                        float bw = tblb.Width * width;
                        float bh = tblb.Height * height;

                        g.TranslateTransform(bx, by);
                        g.RotateTransform((float)(tblb.Angle / Math.PI * 180.0f));
                        g.TranslateTransform(-bx, -by);

                        g.FillEllipse(blbBrush, bx - bw / 2, by - bh / 2, bw, bh);

                        g.TranslateTransform(bx, by);
                        g.RotateTransform(-1 * (float)(tblb.Angle / Math.PI * 180.0f));
                        g.TranslateTransform(-bx, -by);

                        g.DrawString(tblb.BlobID + "", font, fntBrush, new PointF(bx, by));
                    }
                }
            }
        }


    }





    public static void Main(String[] argv)
    {
        int port = 0;
        switch (argv.Length)
        {
            case 1:
                port = int.Parse(argv[0], null);
                if (port == 0) goto default;
                break;
            case 0:
                port = 3333;
                break;
            default:
                Console.WriteLine("usage: mono TuioDemo [port]");
                System.Environment.Exit(0);
                break;
        }

        TuioDemo app = new TuioDemo(port);
        Application.Run(app);
    }

    public void sendSocket(TuioObject markerData)
    {
        try
        {
            // Use selectedPatient variable directly
            if (!string.IsNullOrEmpty(selectedPatient))
            {
                // Prepare the message to send
                string messageToSend = $"Selected Patient: {selectedPatient}";

                // Convert the message to a byte array
                byte[] data = Encoding.UTF8.GetBytes(messageToSend);

                // Send the message to the server
                stream.Write(data, 0, data.Length);
                Console.WriteLine("Sent: {0}", messageToSend);
            }
        }
        catch (Exception e)
        {
            Console.WriteLine("Exception: {0}", e);
        }
    }


    public void receiveSocket()
    {
        try
        {
            // Check if data is available to read
            if (stream != null && stream.DataAvailable)
            {
                // Buffer to store the incoming data
                byte[] data = new byte[256];
                StringBuilder responseData = new StringBuilder();
                int bytes = stream.Read(data, 0, data.Length);

                // Decode the data into a string
                responseData.Append(Encoding.UTF8.GetString(data, 0, bytes));
                Console.WriteLine("Received: {0}", responseData.ToString());

                // Process the received message
                string res = responseData.ToString();

                string[] parts = res.Split(',');

                if (parts.Length == 1)
                {
                    doctor = true;
                }
                else
                {
                    for (int i = 0; i < patients.Count; i++)
                    {
                        if (patients[i].mac == parts[0])
                        {
                            logPatient = patients[i];
                            logPatient.score = parts[1];
                            patient = true;
                            break;
                        }
                    }
                }








            }
        }
        catch (Exception e)
        {
            Console.WriteLine("Exception: {0}", e);
        }
    }

}