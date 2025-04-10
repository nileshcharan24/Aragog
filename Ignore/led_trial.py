from pyfirmata import Arduino
import time

# Connect to Arduino Due on COM4
board = Arduino('COM4')

# Set pin 12 as an output
pin_12 = board.get_pin('d:12:o')

# Blink LED continuously
while True:
    print("on")
    pin_12.write(1)  # Turn LED ON
    time.sleep(1)     # Wait for 1 second
    print("off")
    pin_12.write(0)  # Turn LED OFF
    time.sleep(1)     # Wait for 1 second
