#!/usr/bin/env python3
import argparse
import cv2
from enum import Enum

import time
import numpy as np
import pandas as pd

import pdb
import prey
import robobo
from robobo import SimulationRobobo, HardwareRobobo
from robobo.base import Robobo
import sys
import signal
from typing import Union


class Action(Enum):
    FORWARD = "f"
    BACKWARD = "b"
    ROTATE_L = "rot_L"
    ROTATE_R = "rot_R"


class Player:
    IP: str
    use_sim: bool
    rob: Union[SimulationRobobo, HardwareRobobo]

    def __init__(self, IP="127.0.0.1", use_sim: bool = True):
        self.IP = IP
        self.use_sim = use_sim
        # TODO: add headless arg for sim?
        # TODO: use "speed up simulation" option
        print("connecting...")
        if use_sim:
            tmp = SimulationRobobo().connect(address="127.0.0.1", port=19997)
            if not tmp:
                raise ConnectionError(f"failed to connect to simulation")
            self.rob = tmp
        else:
            # self.rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
            self.rob = HardwareRobobo().connect(address=IP)

        print("\nreseting camera position...")
        # reset camera position
        self.rob.set_phone_tilt(40, 100)

    def play(self):
        print("playing!")
        # if type(self.rob) == SimulationRobobo:
        if isinstance(self.rob, SimulationRobobo):
            self.rob.play_simulation()

        # self.apply_action(Action.FORWARD)
        pdb.set_trace()
        s = self.get_state()

        # when done
        if isinstance(self.rob, SimulationRobobo):
            # pause the simulation and read the collected food
            self.rob.pause_simulation()

            # Stopping the simualtion resets the environment
            self.rob.stop_world()

    def get_state(self, with_img=True, save=False):
        raw_irs = self.rob.read_irs()
        irs = np.log(np.array(raw_irs)) / 10

        img = None
        if with_img:
            img = self.rob.get_image_front()
            if save:
                cv2.imwrite("test_pictures.png", img)
        return {"irs": irs, "img": img}

    def apply_action(
        self, a: Union[Action, str], millis: int = 2000, power: float = 25
    ):
        """
        params:
            dur: duration (ms) of action
            power: float in range [0, 100]
        """
        a = Action(a)
        if a == Action.FORWARD:
            self.rob.move(power, power, millis=millis)
        elif a == Action.BACKWARD:
            self.rob.move(-power, -power, millis=millis)
        elif a == Action.ROTATE_L:
            power = power / 2
            self.rob.move(-power, power, millis=millis)
        elif a == Action.ROTATE_R:
            power = power / 2
            self.rob.move(power, -power, millis=millis)
        else:
            raise NotImplementedError(f"unhandled action {a}")


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():

    # TODO: add headless arg for sim?

    use_sim = True
    IP = "10.15.2.126"  # hardware IP

    signal.signal(signal.SIGINT, terminate_program)

    player = Player(use_sim=use_sim, IP=IP)
    player.play()
    exit(0)

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

    """
    if use_sim:
        rob = robobo.SimulationRobobo().connect(address="127.0.0.1", port=19997)
        rob.play_simulation()
    else:
        # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
        rob = robobo.HardwareRobobo().connect(address=IP)


    # Following code moves the robot
    for i in range(50):
        # print("robobo is at {}".format(rob.position()))
        rob.move(5, 5, millis=2000)
        irs = np.log(np.array(rob.read_irs())) / 10
        print("ROB Irs: {}".format(irs))
        # print("Base sensor detection: ", rob.base_detects_food())

    # print("robobo is at {}".format(rob.position()))
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
    """


if __name__ == "__main__":
    main()
