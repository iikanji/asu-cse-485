from Tkinter import *
from Level import Level

class Categories(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        #Widgets
        label = Label(self, text="Pick Category:")

        button1 = Button(self, text="1. Alphabet", command=lambda:controller.show_frame(Level))
        custButton1 = Button(self, text="Custom 1")

        button2 = Button(self, text="2. Colors and Shapes")
        custButton2 = Button(self, text="Custom 2")

        button3 = Button(self, text="3. Fruits and Vegetables")
        custButton3 = Button(self, text="Custom 3")

        button4 = Button(self, text="4. Feelings")
        custButton4 = Button(self, text="Custom 4")

        button5 = Button(self, text="5. Numbers")

        button6 = Button(self, text="6. Family")

        #Organization
        label.grid(columnspan=2)

        button1.grid(row=1)
        custButton1.grid(row=1, column=1)

        button2.grid(row=2)
        custButton2.grid(row=2, column=1)

        button3.grid(row=3)
        custButton3.grid(row=3, column=1)

        button4.grid(row=4)
        custButton4.grid(row=4, column=1)

        button5.grid(row=5)

        button6.grid(row=6)