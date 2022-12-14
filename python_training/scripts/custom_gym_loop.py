import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import abc, csv, datetime, gym, sys


# Utility function, check existence of a folder
def ensure_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)


class ReinforcementLearning( metaclass=abc.ABCMeta ):


	def __init__(self, env, verbose, str_mod, seed):

			if seed is None: seed = np.random.randint(0, 1000)

			tf.random.set_seed( seed )
			np.random.seed( seed )

			self.env = env
			self.verbose = verbose
			self.run_id = seed
			self.str_mod = str_mod

			self.input_shape = self.env.observation_space.shape
			self.action_space = env.action_space

			self.logging_dir_path = os.getcwd() + '/log'
			self.logging_dir_path += f"/{self.str_mod}" #_seed{self.run_id}"

	def loop( self, num_episodes=10000 ):

		# Initialize the loggers
		#logger_dict = { "reward": [], "success": [], "step": [], "cost": [], "action_list": []  }
		logger_dict = { "reward": [], "success": [], "step": [], "distance": [],"back&forth": [], "spread": [], \
						"sum_of_long_loops":0, "sum_of_fwd_encouragement":0, "sum_of_6_or_more_same_direction_turns":0, "sum_of_back_and_forth":0, \
											 "action_list": [], "cost": []  }

		logger_dict['sum_of_long_loops'] = []
		logger_dict['sum_of_fwd_encouragement'] = []
		logger_dict['sum_of_6_or_more_same_direction_turns'] = []
		logger_dict['sum_of_back_and_forth'] = []

		# Setup the environment for for the I/O on file (logger/models)
		# (only when verbose is active on file)
		if self.verbose > 1:

			# Create the string with the configuration for the file name
			#moved to __init__ self.logging_dir_path += f"/{self.str_mod}_seed{self.run_id}"

			# Extract from the dicitionary of the relevant parameters
			# the parameters to save on the filen name
			# for key, value in self.relevant_params.items():
			# 	self.logging_dir_path += f"_{value}{self.__dict__[key]}"

			# Adding the starting datetime for the file name
			# self.logging_dir_path += '_'
			# self.logging_dir_path += datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

			# Saving the stats in the correct foler
			file_name = self.logging_dir_path + '/run_stats.csv'
			ensure_dir(file_name)

			# Create the CSV file and the writer
			csv_file = open(f"{file_name}", mode='x')
			fieldnames = ['episode', 'reward', 'success', 'step', 'distance', 'back&forth', 'spread', 'cost', 'sum_of_long_loops', 'sum_of_fwd_encouragement', 'sum_of_6_or_more_same_direction_turns', 'sum_of_back_and_forth', 'lambda', 'action_list']
			self.writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
			self.writer.writeheader()

		# Support variable to limit the number of msaved models
		last_saved = 0

		# Iterate the training loop over multiple episodes
		for episode in range(num_episodes):

			# Reset the environment at each new episode
			state = self.env.reset()

			# Initialize the values for the logger
			logger_dict['reward'].append(0)
			logger_dict['success'].append(0)
			logger_dict['step'].append(0)
			logger_dict['cost'].append(0)
			logger_dict['distance'].append( state[-1] )
			logger_dict['back&forth'].append(0)
			logger_dict['spread'].append(0)
			logger_dict['sum_of_long_loops'].append(0)
			logger_dict['sum_of_fwd_encouragement'].append(0)
			logger_dict['sum_of_6_or_more_same_direction_turns'].append(0)
			logger_dict['sum_of_back_and_forth'].append(0)
			logger_dict['action_list'] = []

			# Defining addition metrics
			back_forth = 0
			back_forth_flag = False
			spread = 0

			action_1 = 0
			action_2 = 0
			
			# Main loop of the current episode
			while True:

				# Select the action, perform the action and save the returns in the memory buffer
				action, action_prob = self.get_action(state)
				new_state, reward, done, info = self.env.step(action)

				# Computing additional metrics
				spread = abs((logger_dict['action_list']).count(1) - (logger_dict['action_list']).count(2))
				if len(logger_dict['action_list']) > 0 and action in [1, 2] \
					and logger_dict['action_list'][-1] not in [action, 0]: 
						back_forth += 1
						back_forth_flag = 1
				else:
					back_forth_flag = 0

				# No rules active
				info["cost"] = []
				# Rule 1 active:
				if info['rules'][0]: info['cost'].append(back_forth_flag)
				# Rule 2 active:
				if info['rules'][1]: info['cost'].append(info['sum_of_6_or_more_same_direction_turns'])
				# Rule 5 active:
				if info['rules'][2]: info['cost'].append(info['sum_of_fwd_encouragement'])
				# All active!
				#if all(info['rules']): info['cost'] = ( back_forth_flag + info['sum_of_6_or_more_same_direction_turns'] + info['sum_of_fwd_encouragement'] )

				#
				self.memory_buffer.append([state, action, action_prob, reward, new_state, done, info["cost"]])

				# Update the dictionaries for the logger and the trajectory
				logger_dict['reward'][-1] += reward	
				logger_dict['success'][-1] = 1 if info['goal_reached'] else 0
				logger_dict['step'][-1] += 1	
				logger_dict['cost'][-1] += sum(info['cost'])
				logger_dict['back&forth'][-1] = back_forth
				logger_dict['spread'][-1] = spread
				logger_dict['action_list'].append(action)
				logger_dict['sum_of_long_loops'][-1] += info['sum_of_long_loops']
				logger_dict['sum_of_fwd_encouragement'][-1] += info['sum_of_fwd_encouragement']
				logger_dict['sum_of_6_or_more_same_direction_turns'][-1] += info['sum_of_6_or_more_same_direction_turns']
				logger_dict['sum_of_back_and_forth'][-1] += info['sum_of_back_and_forth']
				
				# Call the update rule of the algorithm
				self.network_update_rule( episode, done )

				# Exit if terminal state and eventually update the state
				if done: break
				state = new_state			

			# Log all the results, depending on the <verbose> parameter
			# here simple print of the results
			if self.verbose > 0:
				last_n =  min(len(logger_dict['reward']), 100)
				reward_last_100 = logger_dict['reward'][-last_n:]
				success_last_100 = logger_dict['success'][-last_n:]
				cost_last_100 = logger_dict['cost'][-last_n:]
				metric_rule_1 = logger_dict['back&forth'][-last_n:]
				metric_rule_2 = logger_dict['sum_of_6_or_more_same_direction_turns'][-last_n:]
				metric_rule_5 = logger_dict['sum_of_fwd_encouragement'][-last_n:]
				
				if not self.avoid_print:
					print( f"({self.str_mod}) Ep: {episode:5}", end=" " )
					print( f"reward: {logger_dict['reward'][-1]:5.2f} (last_100: {np.mean(reward_last_100):5.2f})", end=" " )
					print( f"cost_last_100: {int(np.mean(cost_last_100)):3d}", end=" " )
					print( f"success_last_100 {int(np.mean(success_last_100)*100):4d}%", end=" " )
					print( f"back&forth_last_100 {int(np.mean(metric_rule_1)):3d}", end=" " )
					print( f"6_same_dir_last_100 {int(np.mean(metric_rule_2)):3d}", end=" " )
					print( f"fwd_enco_last_100 {int(np.mean(metric_rule_5)):3d}" )

					# Force output update (server only)
					sys.stdout.flush()
				

			#
			try: lag_val = self.lagrangian_lambda.value().numpy()
			except:	lag_val = 0

			# Log all the results, depending on the <verbose> parameter
			# here save the log to a CSV file
			if self.verbose > 1:	
				self.writer.writerow({ 
					'episode' : episode,
					'reward': logger_dict['reward'][-1], 
					'success': logger_dict['success'][-1], 
					'step': logger_dict['step'][-1], 
					'distance': logger_dict['distance'][-1], 
					'cost': logger_dict['cost'][-1],
					'back&forth': logger_dict['back&forth'][-1], 
					'spread': logger_dict['spread'][-1],
					'sum_of_long_loops': logger_dict['sum_of_long_loops'][-1],
					'sum_of_fwd_encouragement': logger_dict['sum_of_fwd_encouragement'][-1],
					'sum_of_6_or_more_same_direction_turns':logger_dict['sum_of_6_or_more_same_direction_turns'][-1],
					'sum_of_back_and_forth': logger_dict['sum_of_back_and_forth'][-1],
					'lambda': lag_val,
					'action_list': str(logger_dict['action_list'])
				})

			# Log all the results, depending on the <verbose> parameter
			# here save the models generated if "good"
			max_frequency = 2000
			success_rate = int(np.mean(success_last_100) * 100)
			if (success_rate > 98 and episode > 1000 and (episode - last_saved) > max_frequency) or \
				(self.verbose > 2 and success_rate > 95 and episode > 1000 and (episode - last_saved) > max_frequency) or \
				(last_saved == 0 and episode == num_episodes - 1) or ((episode - last_saved) > max_frequency):

				saved_model_path = self.logging_dir_path + '/models'
				ensure_dir(f"{saved_model_path}/test.txt")
				self.actor.save(f"{saved_model_path}/{self.str_mod}_id{self.run_id}_ep{episode}_srate{success_rate}.h5")
				last_saved = episode


	# Class that generate a basic neural netowrk from the given parameters.
	# Can be overrided in the inheriting class for a specific architecture (e.g., dueling)
	def generate_model( self, input_shape, output_size=1, layers=2, nodes=32, last_activation='linear', output_bounds=None ):

		# Fix if the output shape is received as a gym spaces,
		# covnersion to integer
		if isinstance(output_size, gym.spaces.discrete.Discrete): output_size = output_size.n
		if isinstance(output_size, gym.spaces.box.Box): output_size = output_size.shape[0]		

		# Itearte over the provided parametrs to create the network, the input shape must be defined as a multidimensional tuple (e.g, (4,))
		# While the output size as an integer
		hiddens_layers = [tf.keras.layers.Input( shape=input_shape )]
		for _ in range(layers):	hiddens_layers.append( tf.keras.layers.Dense( nodes, activation='relu')( hiddens_layers[-1] ) )
		hiddens_layers.append( tf.keras.layers.Dense( output_size, activation=last_activation)( hiddens_layers[-1] ) )	

		# Normalize the output layer between a range if given,
		# usually used with continuous control for a sigmoid final activation
		if output_bounds is not None: 
			hiddens_layers[-1] = hiddens_layers[-1] * (output_bounds[1] - output_bounds[0]) + output_bounds[0]

		# Create the model with the keras format and return
		return tf.keras.Model( hiddens_layers[0], hiddens_layers[-1] )