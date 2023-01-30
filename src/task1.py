#!/usr/bin/env python3
import argparse
import cv2
from enum import Enum

import time
import numpy as np
import pandas as pd

import math
import pdb
import prey
import robobo
from robobo import SimulationRobobo, HardwareRobobo
from robobo.base import Robobo
import sys
import signal
from typing import Union, Optional, List, Tuple, Dict
import random


class Action(Enum):
    FORWARD = 0
    BACKWARD = 1
    ROTATE_L = 2
    ROTATE_R = 3
    # TODO: define more granular rotation...


class Player:
    """
    TODO: consider making this an abstract base class for all tasks.
    """

    IP: str
    use_sim: bool
    rob: Union[SimulationRobobo, HardwareRobobo]

    def __init__(self, IP="127.0.0.1", use_sim: bool = True):
        self.IP = IP
        self.use_sim = use_sim
        # TODO: add headless arg for sim?
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
        else:
            # self.rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
            self.rob = HardwareRobobo().connect(address=IP)

    def run_episode(
        self,
        epsilon: float = 0.05,
        max_steps: int = 200,
    ) -> Tuple[List, Dict]:
        print("running episode!")
        if isinstance(self.rob, SimulationRobobo):
            # ensure sim is running and reset
            self.toggle_sim(True, hard=True)

        # self.apply_action(Action.FORWARD)
        print("\nresetting camera position...")
        self.rob.set_phone_tilt(0 * math.pi, 100)

        info = {"food_count": 0}
        history = []
        s = self.get_state(save=True)
        self.apply_action(Action.FORWARD, millis=18000)
        self.apply_action(Action.BACKWARD, millis=8000)
        for i in range(max_steps):
            if random.random() < epsilon:
                a = random.choice(list(Action))
            else:
                raise NotImplementedError()
                # a = Action(torch.argmax(qvals).item())

            a = Action.BACKWARD

            print(f"step {i+1}/{max_steps} ({(100* i/max_steps):.2f}%) a={a}")
            print(s["irs"])
            pdb.set_trace()

            self.apply_action(a)
            if isinstance(self.rob, SimulationRobobo):
                info["food_count"] = self.rob.collected_food()

            # get next state
            next_s = self.get_state(save=True)

            r = 0
            # TODO: compute reward
            done = i == max_steps - 1
            history.append(
                {
                    "s": s,
                    "a": a.value,
                    "r": r,
                    "next_s": next_s,
                    "terminal": done,
                }
            )

            # np.isinf(s['irs'][2])
            s = next_s

        # when done
        if isinstance(self.rob, SimulationRobobo):
            # food_count = self.rob.collected_food()
            self.toggle_sim(False)

        return history, info

    def get_state(self, with_img=True, save=False, fname: str = "test_pictures.png"):
        raw_irs = self.rob.read_irs()
        irs = self._process_irs(raw_irs)

        img = None
        if with_img:
            img = self.rob.get_image_front()
            if save:
                cv2.imwrite(fname, img)
        state = {"irs": irs, "img": img}
        if isinstance(self.rob, SimulationRobobo):
            state["pos"] = self.rob.position()
        return state

    @staticmethod
    def _process_irs(raw_irs):
        # return np.log(np.array(raw_irs)) / 10
        return [np.log(x) / 10 if x != False else 999 for x in np.array(raw_irs)]

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
            raise UserWarning("can't toggle sim for HardwareRobobo")

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

    def test_camera(self):
        """Experiment with visualizing effect of different camera angles."""
        print(f"testing camera:")
        if isinstance(self.rob, SimulationRobobo):
            self.toggle_sim(True, hard=True)
        self.apply_action(Action.ROTATE_L)
        self.apply_action(Action.FORWARD)
        self.rob.set_phone_tilt(2 * math.pi, 100)
        self.wait_robot(3000)

        # for k in np.arange(-2.0, 2.0, 1.0 / 30):
        for k in np.arange(0, 1.0, 1.0 / 30):
            print(f"k ={k:.2f}")
            self.rob.set_phone_tilt(math.pi * k, 100)
            self.wait_robot(100)
            s = self.get_state(save=True, fname=f"camera_angles/angle_{k:.2f}_pi.png")

    def test_task3(self):
        """Tests ability of reading position of red puck and green target (base) for task3."""
        print("testing task3")
        assert isinstance(self.rob, SimulationRobobo)

        self.toggle_sim(True, hard=True)
        print("moving...")
        # approach goal:
        self.rob.set_phone_tilt(0.2 * math.pi, 100)
        self.apply_action(Action.FORWARD, millis=2000)
        self.apply_action(Action.ROTATE_R, millis=2800)
        self.apply_action(Action.FORWARD, millis=5000)
        print("done!...")
        while True:
            cur_pos = self.rob.puck_position()
            base_pos = self.rob.base_position()
            detected = self.rob.base_detects_food()
            assert cur_pos is not None and base_pos is not None
            print(f"cur_pos = {cur_pos}")
            print(f"base_pos = {base_pos}")
            print(f"detected = {detected}")
            dist = math.sqrt(
                (cur_pos[0] - base_pos[0]) ** 2 + (cur_pos[1] - base_pos[1]) ** 2
            )
            print(f"dist = {dist:.3f}")

            # a = Action.FORWARD
            self.get_state(save=True)
            pdb.set_trace()
            # self.apply_action(a)
            if detected:
                break
        print("hi")


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

    agent = Player(use_sim=use_sim, IP=IP)
    # agent.report_sim_speed()
    # res = agent.run_episode(epsilon=1.0, max_steps=10)
    agent.test_task3()
    pdb.set_trace()


if __name__ == "__main__":
    main()
