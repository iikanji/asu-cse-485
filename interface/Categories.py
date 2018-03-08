from Tkinter import *

root = Tk() #blank window for widgets to be placed on

#Widgets
label = Label(root, text="Pick Category:")

button1 = Button(root, text="1. Fruit")
custButton1 = Button(root, text="Custom 1")

button2 = Button(root, text="2. Colors")
custButton2 = Button(root, text="Custom 2")

button3 = Button(root, text="3. Food")
custButton3 = Button(root, text="Custom 3")

button4 = Button(root, text="4. Verbs")
custButton4 = Button(root, text="Custom 4")

button5 = Button(root, text="5. Emotions")
custButton5 = Button(root, text="Custom 5")

button6 = Button(root, text="6. Family")
custButton6 = Button(root, text="Custom 6")

custButton7 = Button(root, text="Custom 7")

custButton8 = Button(root, text="Custom 8")

custButton9 = Button(root, text="Custom 9")

custButton10 = Button(root, text="Custom 10")

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
custButton5.grid(row=5, column=1)

button6.grid(row=6)
custButton6.grid(row=6, column=1)

custButton7.grid(row=7, column=1)

custButton8.grid(row=8, column=1)

custButton9.grid(row=9, column=1)

custButton10.grid(row=10, column=1)

root.mainloop() #makes sure window is constantly displayed