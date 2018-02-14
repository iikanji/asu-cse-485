from Tkinter import *

root = Tk() #blank window for widgets to be placed on

#Widgets
pickLabel = Label(root, text="Pick level:")

button1 = Button(root, text="Level 1")
button2 = Button(root, text="Level 2")
button3 = Button(root, text="Level 3")
button4 = Button(root, text="Level 4")
button5 = Button(root, text="Level 5")

cardOrGame = Button(root, text="Back to Play or Cards")
categories = Button(root, text="Back to Categories")


#Organization
pickLabel.pack(fill=X)

button1.pack(fill=X)
button2.pack(fill=X)
button3.pack(fill=X)
button4.pack(fill=X)
button5.pack(fill=X)

cardOrGame.pack(fill=X)
categories.pack(fill=X)


root.mainloop() #makes sure window is constantly displayed