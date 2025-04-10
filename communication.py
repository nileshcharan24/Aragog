from pyfirmata import Arduino
import time 
import tkinter as tk

window = tk.Tk()
window.config(background='black')
window.state('zoomed')
window.title('HC-05 Connection')

board = Arduino('COM4')

pin_12 = board.get_pin('d:12:o')

def light_on():
    pin_12.write(1)

def light_off():
    pin_12.write(0)

button1 = tk.Button(window, text = 'On', font = ('Arial', 40), command = lambda:light_on())
button1.place(x=50, y=50)

window.mainloop()