from random import seed
from tkinter.messagebox import NO
import gym; gym.logger.set_level(40)
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np
import os
import time
from env.robotic_navigation import RoboticNavigation
import collections
import datetime
import random
import argparse

DEBUG_HERE = False

def ensure_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)

class RoboticNavigation_ES_extension(RoboticNavigation):
	"""
	A class that extends RoboticNavigation to incorporate executable-specifications during training
	Extends RoboticNavigation with override_reward method.
	"""

#	def __init__(self, scan_number=7, worker_id=0, env_path="None", random_seed=0):
	def __init__(self, step_limit=300, worker_id=0, editor_run=False, random_seed=0, rules_active=False, rules_list=[]):

		self.actions_deque = collections.deque()
		self.locations_dict = dict()

		self.t_dist_int_last = 0
		self.reset_counter = 0

#		self.reward_deviation = 0.5 #13-4 0.25 #0.5 12-4-22  #12-4-22 1.7
#		self.reward_penalty = -1.3 #13-4 2nd -1.7 #13-4 -0.7 # 12-4-22 -1.7
		self.only_count_asif_penalty = False

		super().__init__(step_limit, worker_id, editor_run, random_seed, rules_active)

		self.reward_penalty = 0
		self.reward_penalty_std_dev = 0
		self.rule1_active = False
		self.rule2_active = False
		self.rule3_active = False
		self.rule4_active = False
		self.rule5_active = False

		self.DO_FIXED = False
		self.DO_UNIFORM = False
		self.DO_NORMAL = False

		if DEBUG_HERE:
			print(rules_list)
		for rule in rules_list:
			key, value = rule.split('=')
			if value is not None:
				setattr(self, key, eval(value))
				if DEBUG_HERE:
					print(key, value)
		if not (self.DO_FIXED or self.DO_NORMAL or self.DO_UNIFORM):
			self.DO_NORMAL = True
		# print(rules_list)
				

	def reset(self):
		self.actions_deque.clear()
		self.locations_dict = dict()

		self.reset_counter += 1
		return super().reset()

	def report_penalty(self, msg, enforce_report=False):
		if enforce_report or self.reset_counter > 0 and self.reset_counter%13 == 0:
			if DEBUG_HERE:
				print(msg)

	def calculate_penalty(self, reward):
		overriden_reward = - self.reward_penalty

		if self.DO_NORMAL:
			overriden_reward = - random.gauss((self.reward_penalty), self.reward_penalty_std_dev)
		elif self.DO_UNIFORM:
			rand = random.uniform(0, 1)
			lower_p = 0.17 #0.33: 9-4-2022
			upper_p = 1 - lower_p
			if rand < lower_p:
				overriden_reward -= self.reward_penalty_std_dev
			elif rand > upper_p:
				overriden_reward += self.reward_penalty_std_dev
		elif self.DO_FIXED:
			overriden_reward = - self.reward_penalty

		return overriden_reward


	def override_reward(self, state, reward, action, done, info ):
		info['sum_of_back_and_forth'] = 0
		info['sum_of_6_or_more_same_direction_turns'] = 0
		info['sum_of_long_loops'] = 0
		info['sum_of_fwd_encouragement'] = 0

		if self.reset_counter < 1:
			return reward

		# If no rules active return reward without penalty
		#rules_active = [self.rule1_active, self.rule2_active, self.rule3_active, self.rule5_active]
		#if not any(rules_active):
		#	return reward

		overriden_reward = reward
		accumulated_penalty = 0  # Added accumulated_penalty to ease rule-mixing

		self.actions_deque.appendleft(action)
		if len(self.actions_deque) >= 3:
			#rule rules below
			if True: #self.rule1_active:
				# scenario 1: do not go back-and-forth, e.g, right->left / left->right / right->left
				# current action is also self.actions_deque[0]
				# exactly right->left->right or left->right->left
				if action != 0 and self.actions_deque[1] != action and self.actions_deque[1] != 0 \
					and self.actions_deque[2] != 0 and self.actions_deque[2] != self.actions_deque[1]:
					# exactly one more right->left->right or left->right->left
					if self.rule1_active and not self.only_count_asif_penalty:
						accumulated_penalty += self.calculate_penalty(reward)
					self.report_penalty(f"{self.actions_deque[2]}->{self.actions_deque[1]}->{action}")
					info['sum_of_back_and_forth'] += 1

			# scenario 2: do not go in circle, e.g., more then 6 consecutive right or left turns
			if len(self.actions_deque) == 7: #and self.rule2_active
				if self.actions_deque.count(1) == 7 or self.actions_deque.count(2) == 7:
					if self.rule2_active and not self.only_count_asif_penalty:
						accumulated_penalty += self.calculate_penalty(reward) #25-5-2022 *5
					self.report_penalty(f"7 consecutive {action}")
					info['sum_of_6_or_more_same_direction_turns'] += 1

				self.actions_deque.pop()

			# scenario 3: do  not go in long circles, e.g., get back to where the same target direction and distance have already
			# seen before
			t_dist = state[-1]
			t_dir = state[-2]
			if t_dist != 0: #and self.rule3_active
				t_dist_int = int(t_dist * 1000)  # as we have floating points here, need to correct. Is 1000 a correct quantization?
				t_dir_int = int(t_dir * 1000)
				if not self.locations_dict or not self.locations_dict.get(t_dist_int) or self.locations_dict[t_dist_int] != t_dir_int:
					self.locations_dict[t_dist_int] = t_dir_int
				# can create [t_dir_int, reset_counter] to get an idea of how long that circle is
				else:
					if self.rule3_active  and not self.only_count_asif_penalty:
						accumulated_penalty += self.calculate_penalty(reward)*6 #13-4 2nd *5 #13-4  #*10 12-4-22 # Need to count #steps in the cycle, instead of "10"
					self.report_penalty(f"long loop !", True)
					info['sum_of_long_loops'] += 1

			# scenario 4: distance to target must decrease over time! DOING NOTHING NOW, believe this is encoded in the PPO algo already
			#gt_ls = ('>' if t_dist_int > self.t_dist_int_last else '<')
			#print("{} {} Last dist {}".format(t_dist_int, gt_ls, self.t_dist_int_last))
			#self.t_dist_int_last = t_dist_int

			# scenario 5: if target is straight ahead, move FWD !
			FWD_DIR = 0.5
			FWD_DIR_TOLERANCE = 0.1
			if action != 0 and (state[3] > 0.35 and state[2] > 0.3 and state[4] > 0.3) and abs(FWD_DIR - t_dir) < FWD_DIR_TOLERANCE: #and self.rule5_active
				# print('Not moving head-on target {}'.format(t_dir))
				if self.rule5_active and not self.only_count_asif_penalty:
					accumulated_penalty += self.calculate_penalty(reward)*0.05 #added on 27-5-2022
				self.report_penalty('Not moving head-on target {}'.format(t_dir), True)
				info['sum_of_fwd_encouragement'] += 1

		if accumulated_penalty != 0: overriden_reward = accumulated_penalty

		info["rules"] = [self.rule1_active, self.rule2_active, self.rule5_active]

		return overriden_reward



