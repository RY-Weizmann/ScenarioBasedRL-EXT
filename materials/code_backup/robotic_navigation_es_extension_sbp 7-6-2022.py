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


from bppy import *
class RoboticNavigation_ES_extension_SBP(RoboticNavigation_ES_extension):
	"""
	A class that extends RoboticNavigation to incorporate executable-specifications using SBP during training
	Extends RoboticNavigation with override_reward method.
	"""

	def __init__(self, step_limit=300, worker_id=0, editor_run=False, random_seed=0, rules_active=False, rules_list=[], kwargs=None):
		super().__init__(step_limit, worker_id, editor_run, random_seed, rules_active, rules_list)
		self.SBP_terminated = False
		self.initial_list = []
		self.SBP_source_name = None
		if not kwargs is None:
			if 'SBP_initial_list' in kwargs.keys():
				self.initial_list = kwargs['SBP_initial_list']
				print(self.initial_list)

		# self.initial_list = [avoid_3_in_a_row(), request_random_action()] #event_sensor(), request_random_action()]
		self.init_bprogram()

		def init_bprogram(self):
			self.b_program = BProgram_ex(bthreads=self.initial_list, source_name=self.SBP_source_name,
										 event_selection_strategy=SimpleEventSelectionStrategy(),
										 listener=None,  # PrintBProgramRunnerListener(),
										 infinite_run=True)
			self.b_program.run(True)
#			self.b_program.initiate_run()
#			self.b_program.superstep_all_bthreads()  # changed on 12-5-2022


	def reset(self):
		self.init_bprogram()
		return super().reset()


	def override_reward(self, reward, action, is_done=False, state_info_dict=None): #before 12-5-2022
		info['sum_of_back_and_forth'] = 0
		info['sum_of_6_or_more_same_direction_turns'] = 0
		info['sum_of_fwd_encouragement'] = 0

		overriden_reward = reward

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
		SBP_requested_ev = self.get_requested_event(action, state_info_dict)
		#  The event is requested externally, thus must check it is not blocked:
		if not self.b_program.is_event_blocked(SBP_requested_ev):
			SBP_terminated = self.b_program.one_pass_all_bthreads(True, SBP_requested_ev)

		assert not SBP_terminated, 'SBP: premature program termination'
		if self.b_program.is_event_blocked(SBP_requested_ev): ###  Mind that either it was already blocked, or became blocked in the call to one_pass_all_bthreads above
			print("Override reward -- Blocked ev {}".format(SBP_requested_ev))
			info['sum_of_back_and_forth'] += 1
			if not self.only_count_asif_penalty:
				overriden_reward = self.reward_penalty
		else:
			SBP_terminated = self.b_program.superstep_all_bthreads()

		return overriden_reward



	def override_reward_super_method(self, state, reward, action, done, info ):
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





if __name__ == "__main__":
import bppy
import random as rand
from bppy import *

iteration_counter = 0
stale_counter = 0  # For testing purpose
stale_action = None  # For testing purpose
def get_random_action():
	global stale_counter
	global stale_action
	act = random.uniform(0, 3)
	random_state = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 5)]

	if act < 1:
		requested_ev = BEvent("SBP_MoveForward")
	elif act < 2:
		requested_ev = BEvent("SBP_TurnRight")
	else:
		requested_ev = BEvent("SBP_TurnLeft")

	requested_ev.data['state'] = random_state
	if abs(act - 1.534) < 0.1:
		stale_counter = 8

	if stale_counter > 0:
		requested_ev = stale_action
		stale_counter -= 1
	else:
		stale_action = requested_ev
	# print(requested_ev)
	return requested_ev


FWD_DIR = 0.5
FWD_DIR_TOLERANCE = 0.1
MINIMAL_CLEARANCE = 0.30
MINIMAL_FWD_CLEARANCE = 0.35
# scenario 5: if target is straight ahead, move FWD !
def SBP_avoid_turning_when_clear():
	blockedEvList = []
	waitforEvList = [BEvent("SBP_MoveForward"), BEvent("SBP_TurnLeft"), BEvent("SBP_TurnRight")]
	while True:
		lastEv = yield {waitFor: waitforEvList, block: blockedEvList}
		# print("{} in SBP_avoid_turning_when_clear iteration: {}".format(lastEv.name, iteration_counter))
		if False and lastEv is None:
			print("None EV in SBP_avoid_turning_when_clear {iteration_counter}")
		else:
			state = lastEv.data['state']
			t_dir = state[-2]
			if state[3] > MINIMAL_FWD_CLEARANCE and state[2] > MINIMAL_CLEARANCE and state[4] > MINIMAL_CLEARANCE and abs(FWD_DIR - t_dir) < FWD_DIR_TOLERANCE: #all clear, goal ahead
				print("SBP_avoid_turning_when_clear dir: {} Blocking {} {} iteration: {}".format(t_dir, BEvent("SBP_TurnLeft"), BEvent("SBP_TurnRight"), iteration_counter))
				blockedEvList.extend([BEvent("SBP_TurnLeft"), BEvent("SBP_TurnRight")])
			else:
				blockedEvList = []
# The Python implementation of a scenario that blocks rotating if the goal is head-on and the path is clear.


def SBP_avoid_k_consecuative_turns():
	k = 7
	counter = 0
	prevEv = None
	blockedEvList = []
	waitforEvList = [BEvent("SBP_MoveForward"), BEvent("SBP_TurnLeft"), BEvent("SBP_TurnRight")]
	while True:
		lastEv = yield {waitFor: waitforEvList, block: blockedEvList}
		# print("{} in SBP_avoid_k_consecuative_turns iteration: {}".format(lastEv.name, iteration_counter))
		if False and lastEv is None:
			print("None in SBP_avoid_k_consecuative_turns iteration: {}".format(iteration_counter))
		else:
			if prevEv is None or lastEv.name == "SBP_MoveForward" or prevEv.name != lastEv.name:
				prevEv = lastEv
				counter = 0
				blockedEvList = []
			else:
				if counter == k - 1:
					# Blocking!
					print("SBP_avoid_k_consecuative_turns Blocking {}, iteration: {}".format(lastEv, iteration_counter))
					blockedEvList.append(lastEv)
				else:
					counter += 1
# The Python implementation of a scenario that blocks rotating in the same direction more then k − 1 consecutive times.


def SBP_avoidBackAndForthRotation():
	blockedEvList = []
	waitforEvList = [BEvent("SBP_MoveForward", None),
					 BEvent("SBP_TurnLeft", None),
					 BEvent("SBP_TurnRight", None)]
	while True:
		lastEv = yield {waitFor: waitforEvList, block: blockedEvList}
		# print("{} in SBP_avoidBackAndForthRotation iteration: {}".format(lastEv.name, iteration_counter))
		if False and lastEv is None:
			print("None in SBP_avoidBackAndForthRotation iteration: {}".format(iteration_counter))
		else:
			if lastEv.name != "SBP_TurnLeft" and lastEv.name != "SBP_TurnRight":
				blockedEvList = []
			else:
				blocked_ev = BEvent("SBP_TurnRight", None) if lastEv.name == "SBP_TurnLeft" else BEvent("SBP_TurnLeft", None)
				blocked_ev.data = None
				# Blocking!
				print("SBP_avoidBackAndForthRotation blocking {}, iteration: {}".format(blocked_ev, iteration_counter))
				blockedEvList.append(blocked_ev)


if __name__ == "__main__":
	initial_list = [SBP_avoid_turning_when_clear(), SBP_avoid_k_consecuative_turns(), SBP_avoidBackAndForthRotation()]
	b_program = BProgram_ex(bthreads=initial_list, #source_name='__main__',
							event_selection_strategy=SimpleEventSelectionStrategy(),
							listener=PrintBProgramRunnerListener(),
							infinite_run=True)

	b_program.run(True)

	while True:
		SBP_requested_ev = get_random_action()
		iteration_counter += 1

		#  The event is requested externally, thus must check it is not blocked:
		if not b_program.is_event_blocked(SBP_requested_ev):
			SBP_terminated = b_program.one_pass_all_bthreads(True, SBP_requested_ev)

		assert not SBP_terminated, 'SBP: premature program termination'
		if b_program.is_event_blocked(SBP_requested_ev): ###  Mind that either it was already blocked, or became blocked in the
														 ###  call to one_pass_all_bthreads above
			print("In mainloop -- Blocked ev {}, iteration: {}".format(SBP_requested_ev, iteration_counter))
		else:
			SBP_terminated = b_program.superstep_all_bthreads()
		print("fininshed super-step {} in mainloop \n\n".format(iteration_counter))
