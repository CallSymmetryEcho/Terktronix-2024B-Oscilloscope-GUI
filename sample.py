#!/usr/bin/python3
"""This is a test script."""

from nwm import *

#controller.initialize()

controller.set_x_bias(5)
controller.test()

"""
stage.initialize()
dlp.initialize()

stage.set_origin_xy()
stage.move_xy(4,5)
controller.show_gui()


dlp.calibrate()
dlp.circle(0,0,6)

controller.show_gui()
location = controller.get_click()
controller.start_tracking()

controller.start_recording(filename)

for voltage in [1,5,10,20]:
    controller.set_x_bias(voltage)
    time.sleep(5000)

controller.stop_recording(filename)

stage.close()
controller.close()
dlp.close()
"""
