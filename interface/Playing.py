from Tkinter import *

root = Tk() #blank window for widgets to be placed on

#Widgets
topFrame = Frame(root)
bottomFrame = Frame(root)

playLabel = Label(topFrame, text="You are playing", fg="red")
imageLabel = Label(topFrame, text="Pretend this is an image")

cardOrGame = Button(bottomFrame, text="Back to Play or Cards")
categories = Button(bottomFrame, text="Back to Categories")


#Organization
topFrame.pack()
bottomFrame.pack(side=BOTTOM)

playLabel.pack()
imageLabel.pack(side=BOTTOM)

cardOrGame.pack()
categories.pack()


root.mainloop() #makes sure window is constantly displayed