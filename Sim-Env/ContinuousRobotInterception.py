# -*- coding: utf-8 -*-

from math import sqrt, pow, atan, asin, sin, pi, cos, tau, remainder
import numpy as np
import random
import gym
from gym import spaces
from gym.envs.classic_control import rendering
from gym.utils.renderer import Renderer
import matplotlib.pyplot as plt
from gym.utils import seeding
from gym.error import DependencyNotInstalled
from typing import Optional
import pygame


class TwoWheelRobotContinuousMovingEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 2
    }

    def __init__(self, render_mode: Optional[str] = None):
        # size of the simulation env
        self.env_x = 600
        self.env_y = 400
        # size of the camera view range
        self.img_width = 320
        self.img_height = 200
        # visualizing
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        self.screen_env = None
        self.screen_cam = None

        # self.viewer_env = rendering.Viewer(self.env_x, self.env_y)
        # self.viewer_cam = rendering.Viewer(self.img_width, self.img_height)
        self.view_range = pi / 2  # the view range of the camera on robot (degree)
        self.r = 20  # radius of the target. To simulate yolo, the shape of the target is assumed as a cylinder
        self.r0 = 10  # radius of robot
        self.middle = 0
        # Action space
        self.action_space = spaces.Box(low=np.array([-1, -1], dtype=float), high=np.array([1, 1], dtype=float), )
        # Observation/State Space
        self.low = np.array([0, 0, -int(self.img_width / 2), 0], dtype=int)
        self.high = np.array([self.img_width, self.img_width, int(self.img_width / 2), self.img_height], dtype=int)
        self.observation_space = spaces.Box(self.low, self.high, dtype=int)
        self.seed()
        self.reset()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.renderer.reset()
        self.renderer.render_step()
        # robot initial orientation
        self.angle_target = pi
        self.distance = 0
        self.task_complete = float(False)
        self.task_half = float(False)
        self.task_failed = float(False)
        self.rewards = []
        # xt = 1
        # yt = 1

        while self.distance <= (self.r + self.r0) or self.angle_target >= self.view_range:
            x = random.randint(0, self.env_x)
            y = random.randint(0, self.env_y)
            yaw_deg = random.randint(0, 360)
            yaw = yaw_deg * pi / 180
            self.robot = np.array([x, y, yaw])
            self.distance = random.uniform(self.r + self.r0, np.max([x, y, self.env_x - x, self.env_y - y]) - self.r0)
            self.theta_target = (random.randint(yaw_deg - 45, yaw_deg + 45) % 360) * pi / 180
            xt = x + self.distance * cos(self.theta_target)
            yt = y + self.distance * sin(self.theta_target)
            self.target = np.array([xt, yt])
            # view angle of target object in camera
            self.angle_target = 2 * (asin(self.r / self.distance) % tau)

        # if the part of the target is in range view
        target_up = (self.theta_target + self.angle_target / 2) % tau
        target_down = (self.theta_target - self.angle_target / 2) % tau
        rob_up = (yaw + pi / 4) % tau
        rob_down = (yaw - pi / 4) % tau
        if (rob_up - target_up) % tau <= pi / 2:
            in_up = True
            up_bound = target_up
        else:
            in_up = False

        if (target_down - rob_down) % tau <= pi / 2:
            in_down = True
            down_bound = target_down
        else:
            in_down = False

        if not in_up and not in_down:
            up_bound = rob_up
            down_bound = rob_up
        elif (not in_up) and in_down:
            up_bound = rob_up
        elif (not in_down) and in_up:
            down_bound = rob_down

        target_middle = (((up_bound - down_bound) % tau) / 2 + down_bound) % tau
        bbox_middle = int(self.img_width * ((rob_up - target_middle) % tau) / self.view_range)
        bbox_width = int(self.img_width * ((up_bound - down_bound) % tau) / self.view_range)
        bbox_bias = int(self.img_width / 2 - bbox_middle)

        bbox_height = int(self.angle_target * self.img_height / self.view_range)
        self.state = np.array([bbox_middle, bbox_width, bbox_bias, bbox_height]).astype(int)
        self.last_state = self.state
        self.middle = 0
        self.step_n = 0
        self.time_out = float(False)

        return self.state

    def step(self, action: int):

        self.step_n += 1
        x, y, yaw = self.robot
        vl = action[0]
        vr = action[1]
        x += cos(yaw) * (vl + vr) / 2
        y += sin(yaw) * (vl + vr) / 2
        yaw += - (vl - vr) / (2 * self.r)
        yaw = yaw % tau
        # x = x % self.env_x
        # y = y % self.env_y


        if x >= self.env_x or y >= self.env_y:
            self.crash = True

        self.robot = np.array([x, y, yaw])

        xt, yt = self.target

        self.distance = sqrt(pow(x - xt, 2) + pow(y - yt, 2))

        if xt >= x and yt >= y:
            self.theta_target = asin((yt - y) / self.distance)
        elif xt >= x and yt < y:
            self.theta_target = 2 * pi - asin(abs(yt - y) / self.distance)
        elif xt < x and yt >= y:
            self.theta_target = pi - asin(abs(yt - y) / self.distance)
        else:
            self.theta_target = pi + asin(abs(yt - y) / self.distance)
        # view angle of target object in camera
        self.angle_target = 2 * (asin(self.r / self.distance) % tau)
        # if the part of the target is in range view
        target_up = (self.theta_target + self.angle_target / 2) % tau
        target_down = (self.theta_target - self.angle_target / 2) % tau
        rob_up = (yaw + pi / 4) % tau
        rob_down = (yaw - pi / 4) % tau
        if (rob_up - target_up) % tau <= pi / 2:
            in_up = True
            up_bound = target_up
        else:
            in_up = False

        if (target_down - rob_down) % tau <= pi / 2:
            in_down = True
            down_bound = target_down
        else:
            in_down = False

        if not in_up and not in_down:
            up_bound = rob_up
            down_bound = rob_up
        elif (not in_up) and in_down:
            up_bound = rob_up
        elif (not in_down) and in_up:
            down_bound = rob_down

        target_middle = (((up_bound - down_bound) % tau) / 2 + down_bound) % tau
        self.target_angle = (up_bound - down_bound) % tau
        bbox_middle = int(self.img_width * ((rob_up - target_middle) % tau) / self.view_range)
        bbox_width = int(self.img_width * ((up_bound - down_bound) % tau) / self.view_range)
        bbox_bias = int(self.img_width / 2 - bbox_middle)
        bbox_height = int(self.angle_target * self.img_height / self.view_range)
        done = False

        if bbox_height >= 0.8 * self.img_height:
            if abs(bbox_bias) <= 0.01 * self.img_width:
                self.task_complete = float(True)
            else:
                self.task_half = float(True)
            done = True

        elif abs(bbox_bias) >= self.img_width / 2: #or bbox_height == 0:
            self.task_failed = float(True)
            done = True
        else:
            done = False
        complete_reward = self.task_complete * 100# 100
        aux_reward = - np.float64(sqrt((2 * bbox_bias/self.img_width) ** 2 + ((self.img_height - bbox_height)/self.img_height) ** 2))
        half_reward = -1 * self.task_half
        time_penalty = - (self.step_n / 1000) ** 2
        failed_penalty = -1000 * self.task_failed
        reward = complete_reward + aux_reward  + failed_penalty + half_reward # + time_penalty
        self.state = np.array([bbox_middle, bbox_width, bbox_bias, bbox_height]).astype(int)

        self.last_state = self.state
        self.reward = reward
        return self.state, self.reward, done, {}

    def test_reset(self, robot, target):
        self.robot = robot
        self.target = target

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_action(self, Q, epsilon=0.2):
        if random.random() <= epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(Q[self.state[2] + int(self.img_width / 2)]) - 1
        return action

    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode='human', close=False):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )
        print(self.state)
        if self.screen_env is None:
            pygame.init()
            self.trans = np.array([0, 0, 0])

            self.Wenv = 640
            self.Henv = 440

            print(self.robot, self.target, self.trans)
            print(self.Wenv, self.Henv)


            width = 40 + self.Wenv + self.img_width
            height = 40 + self.Henv + self.img_height
            print(width, height)

            self.cam_origin = (width - self.img_width - 20,
                               int(height / 2 - self.img_height / 2))

            if mode == "human":
                pygame.display.init()
                self.screen_env = pygame.display.set_mode(
                    (width, height)
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.screen_env = pygame.Surface((width, height))


        Black = (0, 0, 0)
        White = (255, 255, 255)
        Blue = (0, 0, 255)
        Red = (255, 0, 0)
        Green = (0, 255, 0)

        x, y, yaw = (np.array(self.robot) + self.trans)
        xt = self.target[0] - self.trans[0]
        yt = self.target[1] - self.trans[1]

        center_t = (xt, yt)
        center_r = (x, y)

        self.screen_env.fill(White)
        # self.screen_cam.fill(White)

        pygame.draw.circle(self.screen_env, color=Blue, center=center_t, radius=self.r)
        pygame.draw.circle(self.screen_env, color=Red, center=center_r, radius=self.r0)

        pygame.draw.line(self.screen_env, Black, (x, y), (x + 100 * cos(yaw), y + 100 * sin(yaw)), width=1)
        pygame.draw.line(self.screen_env, Blue, (x, y), (x + 100 * cos(yaw + pi / 4), y + 100 * sin(yaw + pi / 4)), width=1)
        pygame.draw.line(self.screen_env, Blue, (x, y), (x + 100 * cos(yaw - pi / 4), y + 100 * sin(yaw - pi / 4)), width=1)

        pygame.draw.line(self.screen_env, Green, (x, y), (xt, yt), width=1)
        length = self.distance
        pygame.draw.line(self.screen_env, Green, (x, y),
                         (x + length * cos(self.theta_target + 0.5 * self.target_angle),
                          y + length * sin(self.theta_target + 0.5 * self.target_angle)),
                         width=1)
        pygame.draw.line(self.screen_env, Green, (x, y),
                         (x + length * cos(self.theta_target - 0.5 * self.target_angle),
                          y + length * sin(self.theta_target - 0.5 * self.target_angle)),
                         width=1)

        # pygame.display.update()

        bbox = (self.state[0] - self.state[1]/2 + self.cam_origin[0],
                self.img_height/2 - self.state[3]/2 + self.cam_origin[1],
                self.state[1],
                self.state[3])
        cam = (self.cam_origin[0],
               self.cam_origin[1],
               self.img_width,
               self.img_height)
        pygame.draw.rect(self.screen_env, Red, bbox, width=2)
        pygame.draw.rect(self.screen_env, Black, cam, width=2)
        #


        if mode is 'rgb_array':
            return self.screen_env
        else:
            pygame.display.update()


    def closer(self):
        if self.viewer_env:
            self.viewer_env.close()
            self.viewer_env = None