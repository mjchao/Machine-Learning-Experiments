'''
Created on Jul 3, 2016

@author: mjchao
'''
import numpy as np
from PIL import Image
from PIL import ImageDraw
from Tkinter import Button
from Tkinter import Canvas
from Tkinter import Frame
from Tkinter import N, S, E, W, CENTER
from Tkinter import Tk
import tkMessageBox

from Model import WIDTH, HEIGHT

class MainWindow(object):
    """Main window that allows users to draw a digit and ask the digit classifer to classify it.
    """

    def __init__(self):
        """Sets up the user interface.
        """
        self.main_window_ = Tk()
        self.main_frame_ = Frame(self.main_window_)

        self.canvas_ = Canvas(self.main_window_, width=WIDTH, height=HEIGHT, bg="black")
        self.canvas_.grid(row=0)
        self.img_ = Image.new(mode="L", size=(WIDTH, HEIGHT), color="black")
        self.img_draw_ = ImageDraw.Draw(self.img_)
        self.canvas_.bind("<B1-Motion>", self.OnMouseDragged)
        self.last_x_ = None
        self.last_y_ = None
        self.line_width_ = 2

        self.classify_ = Button(self.main_window_, text="Classify", command=self.Classify)
        self.classify_.grid(row=1, sticky=N+S+E+W)

        self.clear_ = Button(self.main_window_, text="Clear", command=self.Clear)
        self.clear_.grid(row=2, sticky=N+S+E+W)

        self.main_frame_.place(relx=0.5, rely=0.5, anchor=CENTER)

    def OnMouseDragged(self, event):
        """Handles drawing on the canvas.
        
        Args:
            event: the data associated with the drag event
        """
        if self.last_x_ is not None and self.last_y_ is not None:
            self.canvas_.create_line(self.last_x_, self.last_y_, event.x, event.y, fill="white", width=self.line_width_)
            self.img_draw_.line((self.last_x_, self.last_y_, event.x, event.y), fill="white", width=self.line_width_)

        self.last_x_, self.last_y_ = event.x, event.y

    def Classify(self):
        """Classifies the digit drawn on the canvas.
        """
        print "Classifying..."
        pixels = np.zeros((WIDTH*HEIGHT,))
        for y in range(HEIGHT):
            for x in range(WIDTH):
                pixels[y*HEIGHT + x] = self.img_.getpixel((x, y))

        tkMessageBox.showinfo("Classification Result", "The classifier believes you drew a 0")
        print pixels

    def Clear(self):
        """Clears the digit drawn on the canvas.
        """
        self.canvas_.delete("all")
        self.last_x_, self.last_y_ = None, None

    def Show(self):
        """Shows the main frame.
        """
        self.main_frame_.mainloop()


def main():
    main_window = MainWindow()
    main_window.Show()

if __name__ == "__main__": main()