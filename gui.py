from tkinter import *
from tkinter import filedialog
import os
from PIL import ImageTk,Image
import glob
import shutil
from shutil import copy2
import tkinter.messagebox
from threading import Thread as thread
import time


root = Tk()
root.title('Image Upscaling')
root.geometry('600x600')
root.resizable(width=False, height=False)
color = 'gray77'
root.configure(bg=color)

#############################################################Functions##################################################
#class T():
   # def det(self):
       # self.x = Tk()
       # self.x.mainloop()
  #  def det2(self):
       # self.x.destroy()


def open_file():
    root.source = filedialog.askopenfilename(initialdir = "/", title = "select a file", filetypes = (("jpeg files","*.jpg"), ("all files", "*.*")))
    # print (root.source)
    root.destination = r"test"
    copy2(root.source, root.destination, follow_symlinks=True)
    # for jpgfile in glob.iglob(os.path.join(root.source, "*.jpg")):
        # shutil.copy2(jpgfile, root.destination)


def runfile():
    # k = T()
    # ts = thread(target=k.det, args=())
    # ts.start()
    tkinter.messagebox.showinfo('Processing...','This may take few seconds, please wait. Application may freeze during processing.')
    os.system('python ptest.py')
    # k.det2()
    # tkinter.messagebox.showinfo(title="Loading", message="Starting Application")
    # os.system('python test.py')



def openresult():
    os.system(r'start results')


def runrtx():
    tkinter.messagebox.showinfo('Processing...','This may take few seconds, please wait. Application may freeze during processing.')
    os.system('python pnighttest.py')


def openinstructions():
    os.startfile('instructions.txt')


#############################################################Title Window###############################################


upperlabel1 = Label(root, text='Upscaler', bg=color)
upperlabel1.config(font=('Arial', 35))
upperlabel1.place(x=200, y=40)

#############################################################Buttons####################################################

button1 = Button(root, text='Choose image', width=30, height=2, highlightbackground=color, command=lambda: open_file())
button1.place(x=190, y=250)

button2 = Button(root, text='Test', width=30, height=2, highlightbackground=color, command=lambda: runfile())
button2.place(x=190, y=300)

button3 = Button(root, text='Show Result', width=30, height=2, highlightbackground=color, command=lambda: openresult())
button3.place(x=190, y=350)

button4 = Button(root, text='RTX-On', width=30, height=2, highlightbackground=color, command=lambda: runrtx())
button4.place(x=190, y=400)

button5 = Button(root, text='Instructions', width=30, height=2, highlightbackground=color, command=lambda: openinstructions())
button5.place(x=190, y=450)



root.mainloop()

