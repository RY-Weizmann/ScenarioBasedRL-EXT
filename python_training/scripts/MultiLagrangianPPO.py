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


class MultiLagrangianPPO( ReinforcementLearning ):

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

		# [ALL THE LAGRANGIAN Hyperparameters HERE]
		self.train_lambda = False
		self.start_train_lambda = 5000
		self.cost_limit = []

		# 
		self.relevant_params = {
			'gamma' : 'gamma',
			'trajectory_update' : 'tu',
			'critic_epoch' : 'ce',
			'batch_size' : 'cbs',
			'nodes' : 'nod'
		}

		"""
		# Override the default parameters with kwargs
		self.avoid_print = False
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)
		"""
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
		self.lagrangian_lambda = [tf.Variable( 0.0, constraint=lambda z: tf.clip_by_value(z, 0, 1.0) ) for _ in self.cost_limit]
		self.lambda_optimizer = [tf.keras.optimizers.Adam( learning_rate=0.001 ) for _ in self.cost_limit]

		#
		self.memory_buffer = deque( maxlen=self.memory_size )

		#
		self.actor = self.generate_model(self.input_shape, self.action_space, \
			layers=self.layers, nodes=self.nodes, last_activation='softmax')
		self.critic = self.generate_model(self.input_shape, layers=self.layers_critic, nodes=self.nodes_critic)

		#
		self.actor_optimizer = tf.keras.optimizers.Adam()
		self.critic_optimizer = tf.keras.optimizers.Adam()


	def normalize_lagrangians( self ):

		lambda_sum = sum([el.numpy() for el in self.lagrangian_lambda])
		if lambda_sum < 0.5: return

		for i in range(len(self.lagrangian_lambda)):
			self.lagrangian_lambda[i].assign( self.lagrangian_lambda[i] / (2 * lambda_sum) )


	# Mandatory method to implement for the ReinforcementLearning class, decide the 
	# update frequency and some variable update for on/off policy algorithms
	# (i.e., eps_greedy, buffer, ...)
	def network_update_rule( self, episode, terminal ):

		# Sanity check number of lambda
		info = "Mismatch between the cost thresholds and the cost functions"
		assert len(np.array(self.memory_buffer, dtype=object)[:, 6][0]) == len(self.cost_limit), info

		#
		if episode > self.start_train_lambda: self.train_lambda = True

		# Update of the networks for PPO!
		# - Performed every <trajectory_update> episode
		# - clean up of the buffer at each update (on-policy).
		# - Only on terminal state (collect the full trajectory)
		if terminal and episode % self.trajectory_update == 0:
			self.update_networks(np.array(self.memory_buffer, dtype=object))
			if self.train_lambda: self.update_lag_multiplier(np.array(self.memory_buffer, dtype=object))
			self.memory_buffer.clear()


	# Application of the gradient with TensorFlow and based on the objective function
	def update_networks( self, memory_buffer ):

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

	
	#
	def update_lag_multiplier( self, memory_buffer ):

		for idx, (lagrangian, optimizer) in enumerate(zip(self.lagrangian_lambda, self.lambda_optimizer)):
		#
			with tf.GradientTape() as lag_multiplier_tape:

				#
				#
				lagrangian_objective_function = self.lagrangian_objective_function( memory_buffer, lagrangian, idx )
				lagrangian_gradient = lag_multiplier_tape.gradient(lagrangian_objective_function, [lagrangian])
				optimizer.apply_gradients( zip(lagrangian_gradient, [lagrangian]) )

			print( f"\t[cost] lambda_{idx}: {lagrangian.numpy():1.3f}" )
		
		self.normalize_lagrangians()

	
	def lagrangian_objective_function( self, memory_buffer, lagrangian, idx ):

		# Extract values from buffer
		cost = np.array([el[idx] for el in memory_buffer[:, 6]])
		done = np.vstack(memory_buffer[:, 5])

		# For multiple trajectories find the end of each one
		end_trajectories = np.where(done == True)[0]

		# Computation of the log_prob and the sum of the reward for each trajectory.
		# To obtain the probability of the trajectory i need to sum up the values for each single trajectory and multiply 
		# this value for the cumulative reward (no discounted or 'reward to go' for this vanilla implementation).
		trajectory_penalty = []
		counter = 0
		for i in end_trajectories:
			trajectory_penalty.append( sum(cost[counter : i+1]) )#- self.cost_limit )
			counter = i+1
		
		#
		traj_cost = np.mean(trajectory_penalty)

		#
		return tf.multiply( -lagrangian, (traj_cost - self.cost_limit[idx]) )

	
	# Computing the loss function of the critic for the gradient descent procedure,
	# learn to predict the temporal difference formula for the reward
	def critic_objective_function(self, memory_buffer):

		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		reward = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])

		# Compute the target for the critic
		target = self.temporal_difference(reward, new_state, done).numpy()

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

	
	# Computation of the temporal difference value, a montecarlo approach to 
	# approximate the long term reward from a given point based on teh value function (critic)
	def temporal_difference(self, reward, new_state, done): 
		return reward + (1 - done.astype(int)) * self.gamma * self.critic(new_state) 


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
		baseline = self.critic(state)
		advantage = self.temporal_difference(reward, new_state, done) - baseline

		#
		prob = self.actor(state)
		action_idx = [[counter, val] for counter, val in enumerate(action)]
		prob = tf.expand_dims(tf.gather_nd(prob, action_idx), axis=-1)
		r_theta = tf.math.divide(prob, action_prob) #prob/old_prob

		#
		clip_val = 0.2
		obj_1 = r_theta * advantage
		obj_2 = tf.clip_by_value(r_theta, 1-clip_val, 1+clip_val) * advantage
		partial_objective = tf.math.minimum(obj_1, obj_2)

		# Sum over the trajectory (Theoretical implementation)
		partial_objective = tf.math.reduce_sum(partial_objective)

		# Compute the cost penalty for the objective
		cost_objective = 0
		cost_ratio = tf.math.reduce_sum( tf.clip_by_value(r_theta, 1-clip_val, 1+clip_val) ) 
		for idx, _ in enumerate( self.cost_limit ):
			cost_sum = sum([el[idx] for el in memory_buffer[:, 6]])
			cost_objective += -self.lagrangian_lambda[idx].numpy() * (cost_ratio * cost_sum)

		# Compute the combined objective function
		rew_lambda = 1 - sum([el.numpy() for el in self.lagrangian_lambda])
		objective_function = rew_lambda * partial_objective + cost_objective

		#
		return -objective_function

