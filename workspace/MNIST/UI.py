'''
Created on Jul 3, 2016

@author: mjchao
'''
from Tkinter import Button
from Tkinter import Canvas
from Tkinter import Frame
from Tkinter import N, S, E, W, CENTER
from Tkinter import Tk
import tkMessageBox

class MainWindow(object):
    
    def __init__(self):
        self.main_window_ = Tk()
        self.main_frame_ = Frame(self.main_window_)
        
        self.canvas_ = Canvas(self.main_window_, width=32, height=32, bg="black")
        self.canvas_.grid(row=0)
        self.canvas_.bind("<B1-Motion>", self.OnMouseMove)
        self.last_x_ = None
        self.last_y_ = None
        self.line_width_ = 2
        
        self.classify_ = Button(self.main_window_, text="Classify", command=self.Classify)
        self.classify_.grid(row=1, sticky=N+S+E+W)
        
        self.clear_ = Button(self.main_window_, text="Clear", command=self.Clear)
        self.clear_.grid(row=2, sticky=N+S+E+W)
        
        self.main_frame_.place(relx=0.5, rely=0.5, anchor=CENTER)
        
    def OnMouseMove(self, event):
        if self.last_x_ is not None and self.last_y_ is not None:
            self.canvas_.create_line(self.last_x_, self.last_y_, event.x, event.y, fill="white", width=self.line_width_)
        
        self.last_x_, self.last_y_ = event.x, event.y
        
    def Classify(self):
        print "Classifying..."
        tkMessageBox.showinfo("Classification Result", "The classifier believes you drew a 0")
        
    def Clear(self):
        self.canvas_.delete("all")
        self.last_x_, self.last_y_ = None, None
    
    def Show(self):
        self.main_frame_.mainloop()
    
def main():
    main_window = MainWindow()
    main_window.Show()

if __name__ == "__main__": main()