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

using System;
using System.Drawing;
using System.Windows.Forms;
using System.ComponentModel;
using System.Collections.Generic;
using System.Collections;
using System.Threading;
using TUIO;
using System.IO;
using System.Drawing.Drawing2D;
using System.Net.Sockets;
using System.Text;



public class TuioDemo : Form, TuioListener
{
	private TuioClient client;
	private Dictionary<long, TuioObject> objectList;
	private Dictionary<long, TuioCursor> cursorList;
	private Dictionary<long, TuioBlob> blobList;

	public static int width, height;
	private int window_width = 640;
	private int window_height = 480;
	private int window_left = 0;
	private int window_top = 0;
	private int screen_width = Screen.PrimaryScreen.Bounds.Width;
	private int screen_height = Screen.PrimaryScreen.Bounds.Height;
	public int prev_id = -1;
	private bool fullscreen;
	private bool verbose;

	public string serverIP = "localhost"; // IP address of the Python server
	public int port = 8000;               // Port number matching the Python server
	int flag = 0;
	Font font = new Font("Arial", 10.0f);
	SolidBrush fntBrush = new SolidBrush(Color.White);
	SolidBrush bgrBrush = new SolidBrush(Color.FromArgb(0, 0, 64));
	SolidBrush curBrush = new SolidBrush(Color.FromArgb(192, 0, 192));
	SolidBrush objBrush = new SolidBrush(Color.FromArgb(64, 0, 0));
	SolidBrush blbBrush = new SolidBrush(Color.FromArgb(64, 64, 64));
	Pen curPen = new Pen(new SolidBrush(Color.Blue), 1);
	private string objectImagePath;
	private string backgroundImagePath;
	TcpClient client1;
	NetworkStream stream;
	string title = "kena";
	string prevP = "";

	private string selectedPatient;

	public TuioDemo(int port)
	{

		verbose = false;
		fullscreen = false;
		width = window_width;
		height = window_height;

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

				window_left = this.Left;
				window_top = this.Top;

				this.FormBorderStyle = FormBorderStyle.None;
				this.Left = 0;
				this.Top = 0;
				this.Width = screen_width;
				this.Height = screen_height;

				fullscreen = true;
			}
			else
			{

				width = window_width;
				height = window_height;

				this.FormBorderStyle = FormBorderStyle.Sizable;
				this.Left = window_left;
				this.Top = window_top;
				this.Width = window_width;
				this.Height = window_height;

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
		Invalidate();
	}

	protected override void OnPaintBackground(PaintEventArgs pevent)
	{
		// Getting the graphics object
		Graphics g = pevent.Graphics;

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
		int menuRadius = height / 4;  // Adjust size based on your design
		Point center = new Point(width / 2, height / 2);
		g.DrawEllipse(Pens.White, center.X - menuRadius, center.Y - menuRadius, menuRadius * 2, menuRadius * 2);

		// Draw arcs
		for (int i = 0; i < 4; i++)
		{
			double angleStart = i * 90;
			g.FillPie(Brushes.Gray, center.X - menuRadius, center.Y - menuRadius, menuRadius * 2, menuRadius * 2, (float)angleStart, 90);
		}

		// Define patient names based on marker ID
		string[] patientNamesMarker1 = { "Patient A1", "Patient B1", "Patient C1", "Patient D1" };
		string[] patientNamesMarker2 = { "Patient A2", "Patient B2", "Patient C2", "Patient D2" };

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
					g.DrawString("" + tobj.Angle * (180.0 / Math.PI), font, Brushes.White, 0, 0);

					
						// Calculate the angle of the marker
						double angle = Math.Atan2(markerY - center.Y, markerX - center.X) * (180.0 / Math.PI);
						if (angle < 0) angle += 360; // Convert angle to 0-360 range

						// Determine which arc the marker is in
						int selectedQuadrant = (int)(angle / 90) % 4;

						// Highlight the selected quadrant
						g.FillPie(Brushes.Yellow, center.X - menuRadius, center.Y - menuRadius, menuRadius * 2, menuRadius * 2, selectedQuadrant * 90, 90);

						// Display patient names based on the marker ID
						string[] currentPatientNames = tobj.SymbolID == 1 ? patientNamesMarker1 :
														tobj.SymbolID == 2 ? patientNamesMarker2 : null;

						
						

						if (currentPatientNames != null)
						{
							g.DrawString(currentPatientNames[selectedQuadrant], font, Brushes.White,
										 new Point(center.X - menuRadius / 2, center.Y - menuRadius / 2 + selectedQuadrant * 30));
							selectedPatient = currentPatientNames[selectedQuadrant];

							if ((tobj.Angle * (180.0 / Math.PI)  >= 77 && tobj.Angle * (180.0 / Math.PI) < 280))
							{
								if(prevP != selectedPatient)
                                {

									prevP = selectedPatient;
									SendMarkerData(tobj);
									
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

	public void SendMarkerData(TuioObject markerData)
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


}