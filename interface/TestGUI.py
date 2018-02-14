from Tkinter import *

root = Tk() #blank window for widgets to be placed on

def printStatement(event):
    print "Hello World"

#Widgets
button1 = Button(root, text="Print")

button1.bind("<Button-1>", printStatement)

#Organization
button1.pack()

root.mainloop() #makes sure window is constantly displayed