from Tkinter import *

root = Tk() #blank window for widgets to be placed on

#Widgets
learnOrPlay = Label(root, text="Are you going to learn or play?")

flashButton = Button(root, text="Review Flashcards")
gameButton = Button(root, text="Play Game")

returnButton = Button(root, text="Back to Categories")

#Organization
learnOrPlay.pack(fill=X)
flashButton.pack(side="left")
gameButton.pack(side="left")

returnButton.pack()

root.mainloop() #makes sure window is constantly displayed