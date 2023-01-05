#!/usr/bin/env python3
from __future__ import print_function
import argparse

import time
import numpy as np
import pandas as pd

import robobo
import cv2
import sys
import signal
import prey
from enum import Enum

class Action(Enum):
    FORWARD = "forward"
    BACKWARD = "forward"


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def main():
    # TODO: add headless arg for sim?

    use_sim = True
    IP ='10.15.2.126' # hardware IP

    signal.signal(signal.SIGINT, terminate_program)

    if use_sim:
        rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
        rob.play_simulation()
    else:
        # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
        rob = robobo.HardwareRobobo().connect(address=IP)

    # TODO'S:
    # detect collisions (later assign negative reward, reset simulation?)

    # save history of states to disk
    # v0: use clustering to save a discrete set of states

    # example state:
    # ex_state = { irs: [       -inf        -inf        -inf -0.37789082 -0.46607302 -0.63648463 -0.46184335 -0.37318743] }

    # example actions:

    # create initial Q-table, run d

    # train a simple DQN using just IR data (sipmle network)

    # reward: -100 for hitting blocks, -1 per time step?
    #   + reward based on max distanced traveled from furthese point in last 30 sec of history?
    #   or + reward based on speed (if not hitting object)

    # questions:
    #  can you read global position in sim?  (we could reward "exploration" of env grid)
    #  how to detect collsions?

    # part 2: 
    # save (downsampled) samera images, run green mask over them, filter out small blobs

    # train a simple DQN using IR states and sampled


    # Following code moves the robot
    for i in range(50):
        #print("robobo is at {}".format(rob.position()))
        rob.move(5, 5, 2000)
        irs = np.log(np.array(rob.read_irs()))/10
        print("ROB Irs: {}".format(irs))
        #print("Base sensor detection: ", rob.base_detects_food())
   
    #print("robobo is at {}".format(rob.position()))
    # Following code gets an image from the camera

    image = rob.get_image_front()
    # IMPORTANT! `image` returned by the simulator is BGR, not RGB
    cv2.imwrite("test_pictures.png", image)

    time.sleep(0.1)

    if use_sim:
        # pause the simulation and read the collected food
        rob.pause_simulation()
        
        # Stopping the simualtion resets the environment
        rob.stop_world()


if __name__ == "__main__":
    main()
