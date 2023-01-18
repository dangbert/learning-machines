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
from typing import Union, Optional, List
import random


class Action(Enum):
    FORWARD = 0
    BACKWARD = 1
    ROTATE_L = 2
    ROTATE_R = 3
    # TODO: define more granular rotation...


class Player:
    IP: str
    use_sim: bool
    rob: Union[SimulationRobobo, HardwareRobobo]

    def __init__(self, IP="127.0.0.1", use_sim: bool = True):
        self.IP = IP
        self.use_sim = use_sim
        # TODO: add headless arg for sim?
        # TODO: use "speed up simulation" option
        realtime = False
        print("connecting...")
        if use_sim:
            number = ""  # "2"
            # number = "#0"
            tmp = SimulationRobobo(number=number).connect(
                address="127.0.0.1", port=19997, realtime=realtime
            )
            if not tmp:
                raise ConnectionError(f"failed to connect to simulation")
            self.rob = tmp

            # from karine: True means enable real time mode
            # vrep.simxSetBooleanParameter(self._clientID, 25, False, vrep.simx_opmode_oneshot)
            self.report_sim_speed()
            exit(0)
        else:
            # self.rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
            self.rob = HardwareRobobo().connect(address=IP)

    def run_episode(
        self,
        epsilon: float,
        max_steps: int = 200,
    ) -> List:
        print("running episode!")
        if isinstance(self.rob, SimulationRobobo):
            # ensure sim is running and reset
            self.toggle_sim(True, hard=True)

        # self.apply_action(Action.FORWARD)
        print("\nresetting camera position...")
        # reset camera position
        # for angle in range(0, 90, 5):
        # for angle in range(40, 50, 1):
        # for angle in np.arange(40, 50, 0.5):
        #    self.rob.set_phone_tilt(angle, 100)
        #    self.wait_robot(3000)
        #    s = self.get_state(save=True, fname=f"camera_angles/angle_{angle}.png")

        history = []
        s = self.get_state(save=True)
        for i in range(max_steps):
            if random.random() < epsilon:
                a = random.choice(list(Action))
            else:
                raise NotImplementedError()
                # a = Action(torch.argmax(qvals).item())

            print(f"step {i+1}/{max_steps} ({(100* i/max_steps):.2f}%) a={a}")
            print(s["irs"])
            pdb.set_trace()

            # if i % 10 == 0:
            self.apply_action(a)

            # get next state
            s = self.get_state(save=True)
            # np.isinf(s['irs'][2])

        # when done
        if isinstance(self.rob, SimulationRobobo):
            self.toggle_sim(False)
        return history

    def get_state(self, with_img=True, save=False, fname: str = "test_pictures.png"):
        raw_irs = self.rob.read_irs()
        irs = np.log(np.array(raw_irs)) / 10

        img = None
        if with_img:
            img = self.rob.get_image_front()
            if save:
                cv2.imwrite(fname, img)
        state = {"irs": irs, "img": img}
        # if isinstance(self.rob, SimulationRobobo):
        #     state["pos"] = self.rob.base_position()
        return state

    def apply_action(
        self,
        a: Union[Action, str],
        millis: int = 2000,
        power: float = 25,
        wait: bool = True,
    ):
        """
        params:
            dur: duration (ms) of action
            power: float in range [0, 100]
            wait: whether to wait for action to finish before returning
        """
        print(f"applying action: {a}")
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
        if wait:
            self.wait_robot(millis)

    def wait_robot(self, millis: float):
        """Wait for desired milliseconds (in real time for HardwareRoboo, or sim time otherwise)"""
        if isinstance(self.rob, HardwareRobobo):
            return time.sleep(millis / 1000)

        start_time = self.rob.get_sim_time()
        if start_time == 0:
            raise UserWarning(f"unable to wait for robot: robot is stopped")
        while True:
            time.sleep(0.005)
            if self.rob.get_sim_time() - start_time >= millis:
                # print(f"waited {self.rob.get_sim_time() - start_time}")
                return

    def toggle_sim(self, on: bool, hard: bool = False):
        """
        Toggle sim on or off (if needed based on current sim state).
        If stopping the sim, we wait to return until the sim is known to have finished stopping.

        params:
            on: whether sim should be running
            hard: hard reset sim (i.e. ensuring world is reset before playing)
        """
        if isinstance(self.rob, HardwareRobobo):
            raise UserWarning("can't toggle sim from HardwareRobobo")

        if hard:
            self.toggle_sim(on)
            self.toggle_sim(not on)
            self.toggle_sim(on)

        running = self.rob.is_simulation_running() == 1
        sim_state = (
            self.rob.check_simulation_state()
        )  # seems to return 3 for paused, 0 for stopped/off, 1 for on
        print(f"ensuring state on={on} (current state = {sim_state})")
        if on:
            if not running:
                print(f"starting sim...")
                self.rob.play_simulation()
            return

        if running:
            print(f"stopping sim...")
            # pause the simulation and read the collected food
            # self.rob.pause_simulation()
            self.rob.stop_world()
            while self.rob.is_simulation_running() != 0:
                time.sleep(0.05)

    def report_sim_speed(self):
        """Measures and reports how fast the sim is relative to real time."""
        if isinstance(self.rob, HardwareRobobo):
            return

        print(f"measuring sim speed...")
        was_running = self.rob.is_simulation_running() == 1
        if not was_running:
            self.toggle_sim(True)

        start_time = self.rob.get_sim_time()
        dur = 1500
        time.sleep(dur / 1000)
        end_time = self.rob.get_sim_time()

        ratio = (end_time - start_time) / dur
        print(f"real time ratio = {ratio:.3f}")
        if not was_running:
            self.toggle_sim(False)
        return ratio


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():

    # TODO: add headless arg for sim?
    # https://www.coppeliarobotics.com/helpFiles/en/commandLine.htm
    #   ./vrep -h ../assets/arena_obstacles.ttt
    # see also how to start vrep from python
    #   https://github.com/nikhil3456/VrepLibrary

    use_sim = True
    IP = "10.15.2.126"  # hardware IP

    signal.signal(signal.SIGINT, terminate_program)

    player = Player(use_sim=use_sim, IP=IP)
    player.run_episode(epsilon=1.0)
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
