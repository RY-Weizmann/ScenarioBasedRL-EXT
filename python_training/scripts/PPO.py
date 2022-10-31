from scripts.custom_gym_loop import ReinforcementLearning
from collections import deque
import numpy as np
import tensorflow as tf
import os
import datetime

def ensure_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)

class PPO( ReinforcementLearning ):

	"""
	Class that inherits from ReinforcementLearning to implements the PPO algorithm, the original paper can be found here:
	https://arxiv.org/abs/1707.06347 [1]

	[1] Proximal Policy Optimization Algorithms, 
		Schulman et al., 
		arXiv preprint arXiv:1707.06347, 2017

	"""


	# Constructor of the class
	def __init__( self, env, verbose, str_mod="PPO", seed=None, **kwargs ):

		#
		super().__init__( env, verbose, str_mod, seed )

		#
		tf.random.set_seed( seed )
		np.random.seed( seed )

		# Hyperparameters
		self.memory_size = None
		self.gamma = 0.99
		self.trajectory_update = 10
		self.critic_epoch = 40
		self.batch_size = 128
		self.layers = 2
		self.nodes = 32
		self.layers_critic = self.layers
		self.nodes_critic = self.nodes

		# Approach modifiers
		self.trajectory_mean = False
		self.montecarlo = False

		# 
		self.relevant_params = {
			'gamma' : 'gamma',
			'trajectory_update' : 'tu',
			'critic_epoch' : 'ce',
			'batch_size' : 'cbs',
			'nodes' : 'nod'
		}

		#for key, value in self.relevant_params.items():
		#	self.logging_dir_path += f"_{value}{self.__dict__[key]}"

		self.logging_dir_path += '_'
		self.logging_dir_path += datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

		f_name = self.logging_dir_path + "/kwargs.txt"
		ensure_dir(f_name)
		f = open(f_name, "a")
		f.write(str(kwargs))
		f.close()
		#TODO: dump kwargs to standard file name: "kwargs.txt" or alike

		# Override the default parameters with kwargs
		### TODO Handle avoid_print here !!!
		self.avoid_print = False
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		#
		self.memory_buffer = deque( maxlen=self.memory_size )

		#
		self.actor = self.generate_model(self.input_shape, self.action_space, \
			layers=self.layers, nodes=self.nodes, last_activation='softmax')
		self.critic = self.generate_model(self.input_shape, layers=self.layers_critic, nodes=self.nodes_critic)

		#
		self.actor_optimizer = tf.keras.optimizers.Adam()
		self.critic_optimizer = tf.keras.optimizers.Adam()


	# Mandatory method to implement for the ReinforcementLearning class, decide the 
	# update frequency and some variable update for on/off policy algorithms
	# (i.e., eps_greedy, buffer, ...)
	def network_update_rule( self, episode, terminal ):

		# Update of the networks for PPO!
		# - Performed every <trajectory_update> episode
		# - clean up of the buffer at each update (on-policy).
		# - Only on terminal state (collect the full trajectory)
		if terminal and episode % self.trajectory_update == 0:
			self.update_networks(np.array(self.memory_buffer, dtype=object))
			self.memory_buffer.clear()


	# Application of the gradient with TensorFlow and based on the objective function
	def update_networks( self, memory_buffer ):

		# Apply penalties
		reshaped_rules = np.stack(memory_buffer[:, 6], axis=0)
		new_cost = np.dot(reshaped_rules, [0.0, 0.0, 0.0])
		memory_buffer[:, 3] = memory_buffer[:, 3] - new_cost

		# 
		if self.montecarlo:
			memory_buffer[:, 3] = self.discount_reward( memory_buffer[:, 3], memory_buffer[:, 5] )

		# Critic update (repeated epoch times on a batch, fixed):
		for _ in range( self.critic_epoch ):

			# Computing a random sample of elements from the batch for the training,
			# randomized at each epoch
			idx = np.random.randint(memory_buffer.shape[0], size=self.batch_size)
			training_batch = memory_buffer[idx]

			#
			with tf.GradientTape() as critic_tape:

				# Compute the objective function, compute the gradient information and apply the
				# gradient with the optimizer
				critic_objective_function = self.critic_objective_function( training_batch )
				critic_gradient = critic_tape.gradient( critic_objective_function, self.critic.trainable_variables )
				self.critic_optimizer.apply_gradients( zip(critic_gradient, self.critic.trainable_variables) )

		# Actor update (repeated 1 time for each call):
		with tf.GradientTape() as actor_tape:

			# Compute the objective function, compute the gradient information and apply the
			# gradient with the optimizer
			actor_objective_function = self.actor_objective_function( memory_buffer )
			actor_gradient = actor_tape.gradient(actor_objective_function, self.actor.trainable_variables)
			self.actor_optimizer.apply_gradients( zip(actor_gradient, self.actor.trainable_variables) )


	# Computation of the temporal difference value, a montecarlo approach to 
	# approximate the long term reward from a given point based on teh value function (critic)
	def temporal_difference(self, reward, new_state, done): 
		return reward + (1 - done.astype(int)) * self.gamma * self.critic(new_state) 

	#
	#
	def discount_reward(self, reward, done):

		# For multiple trajectories find the end of each one
		end_trajectories = np.where(done == True)[0]
		
		full_trajectories_reward = []
		start_traj = -1

		for end_traj in end_trajectories:

			sum_rewards = 0
			trajectory_rewards = []
			for i in range(end_traj, start_traj, -1):
				sum_rewards = reward[i] + self.gamma * sum_rewards 
				trajectory_rewards.append( sum_rewards )
			trajectory_rewards.reverse()

			#
			full_trajectories_reward += trajectory_rewards.copy() 
			start_traj = end_traj

		#
		return np.array(full_trajectories_reward)

	
	# Computing the loss function of the critic for the gradient descent procedure,
	# learn to predict the temporal difference formula for the reward
	def critic_objective_function(self, memory_buffer):

		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		reward = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])

		# Compute the target for the critic
		# (for the MC case notice that the reward is already discounted)
		if not self.montecarlo:
			target = self.temporal_difference(reward, new_state, done).numpy()
		else:
			target = reward

		# The loss function here is a simple gradient descent to train the
		# critic to predict the real value obtained from the temporal difference call
		# with a standard MSE loss function
		predicted_value = self.critic(state)
		mse = tf.math.square(predicted_value - target)

		#
		return tf.math.reduce_mean(mse)


	# Mandatory method to implement for the ReinforcementLearning class
	# here we select thea action based on the state, for policy gradient method we obtain
	# a probability from the network, from where we perform a sampling
	def get_action(self, state):
		softmax_out = self.actor(state.reshape((1, -1)))
		selected_action = np.random.choice(self.action_space.n, p=softmax_out.numpy()[0])
		return selected_action, softmax_out[0][selected_action]


	# Computing the objective function of the actor for the gradient ascent procedure,
	# here is where the 'magic happens'.
	# Note that in PPO (both mc and TD) the return is now the advantage instead to the
	# cumulative trajectory reward of REINFORCE. Now it does not make sense to consider 
	# multiple trajectories for the sum and then computing the mean, we can directly consider 
	# a unique trakectories and compute the sum at the final stage.
	# In this context the kind of rollout (mc or TD) does not make any changes, the approach 
	# is still actor critic and the return is a 'single step' advantage.
	def actor_objective_function( self, memory_buffer ):

		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = memory_buffer[:, 1]
		action_prob = np.vstack(memory_buffer[:, 2])
		reward = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])	

		# Computation of the advantege
		# (for the MC case notice that the reward is already discounted)
		if not self.montecarlo:
			baseline = self.critic(state)
			advantage = self.temporal_difference(reward, new_state, done) - baseline
		else:
			baseline = self.critic(state)
			advantage = reward - baseline

		#
		prob = self.actor(state)
		action_idx = [[counter, val] for counter, val in enumerate(action)] #Trick to obatin the coordinates of each desired action
		prob = tf.expand_dims(tf.gather_nd(prob, action_idx), axis=-1)
		r_theta = tf.math.divide(prob, action_prob) #prob/old_prob

		#
		clip_val = 0.2
		obj_1 = r_theta * advantage
		obj_2 = tf.clip_by_value(r_theta, 1-clip_val, 1+clip_val) * advantage
		partial_objective = tf.math.minimum(obj_1, obj_2)

		# Mean over the trajectory (OpeanAI implementation)
		if self.trajectory_mean: return -tf.math.reduce_mean(partial_objective)

		# Sum over the trajectory (Theoretical implementation)
		return -tf.math.reduce_sum(partial_objective)


	######################################################
	##### NOT IMPLEMENTED YET - TEMPORARY EXPERIMENTS ####
	######################################################
	#
	#def trajectory_penalized( self, memory_buffer ):

		#gradient_scale = 0.2 if rules_violated else 1
		#actor_gradient = [ tf.math.scalar_mul(gradient_scale, nd_array) for nd_array in actor_gradient ]

		# RULES: update expected advantage and add penalty
		#if self.reward_penalized( memory_buffer, advantage ): 
		#	print( f"\t[rules] active with penalty of {self.rules_expected_advantage:0.4f}" )
		#	advantage -= np.abs(advantage) #self.rules_expected_advantage / len(advantage)

		# RULES: compute partial advantage
		#trajectory_penalty = 0
		#if self.reward_penalized( memory_buffer ):
		#	print("\t[RUL]Trajectory with penalties...")
		#	trajectory_penalty = tf.math.reduce_sum(prob) * (-1)#self.rules_expected_penalty

		#
		#advantage_sum = sum(advantage.numpy().flatten().tolist()) 
		#self.rules_expected_advantage = (self.rules_expected_advantage * 0.1) + (advantage_sum * 0.9)

		#
		#done = np.vstack(memory_buffer[:, 5])	
		#rules_modifiers = memory_buffer[:, 6]

		#
		#end_trajectories = np.where(done == True)[0]
		#return bool(max(rules_modifiers[end_trajectories]))