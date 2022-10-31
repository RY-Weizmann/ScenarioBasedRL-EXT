import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import glob, shutil, scipy, csv
import tensorflow as tf
import numpy as np
import logging; logging.getLogger("tensorflow").setLevel( logging.ERROR )

def get_action( state, policy ):

	input_state = np.array(state).reshape((1, -1))
	softmax_out = policy( input_state ).numpy()
	selected_action = np.argmax( softmax_out )

	# Computing pre-activation values
	intermediate_layer_model = tf.keras.models.Model(inputs=policy.input, outputs=policy.layers[-1].input)
	W, a = policy.layers[-1].get_weights()
	pre_activation = (intermediate_layer_model( input_state ).numpy().dot( W )) + a
	pre_activation = pre_activation[0]

	return selected_action, softmax_out[0], pre_activation


def get_counter_example( policy_network, input_point, target, min_bound=[], max_bound=[], step=400, eps=0.001 ):

	# 
	input_point = np.array( [input_point] )
	input_size = input_point.shape[1]

	# Create the fake input with 1 node, and add a layer
	# that simulate the input point
	fake_model = tf.keras.models.Sequential()
	fake_model.add( tf.keras.layers.Input( shape=1 ) )

	fake_model.add( tf.keras.layers.Dense( 

		input_size, 
		name="fake_input", 
		use_bias=False ,
		kernel_constraint = tf.keras.constraints.MinMaxNorm( min_value=min_bound,
															 max_value=max_bound
							)

	) )

	for l in policy_network.layers[1:]:	fake_model.add( l )

	# Set the weights for the first layer (i.e., input point)
	fake_model.layers[0].set_weights( [input_point] )

	# Set all the variable not trianable
	for layer in fake_model.layers[1:]: layer.trainable = False

	# Set the optimizer and the loss function
	loss = tf.keras.losses.CategoricalCrossentropy()
	optimizer = tf.keras.optimizers.SGD( learning_rate=eps )
	fake_model.compile( loss=loss, optimizer=optimizer )

	new_input_list = []

	for _ in range( 50 ):

		fake_model.fit(
			np.array([[1]]), 
			np.array([target]),
			epochs=1,
			verbose=0
		)

		new_input = fake_model.layers[0].get_weights()[0][0].reshape(-1, input_size)
		new_input_list.append( new_input )

	return new_input_list


def generate_prp_bounds( prp ):

	prp_bounds = [
		[0.2, 1],
		[0.2, 1],
		[0.2, 1],
		[0.2, 1],
		[0.2, 1],
		[0.2, 1],
		[0.2, 1],
		[  0, 1], # Heading
		[0.2, 1] # Distance
	]

	if prp == 1: prp_bounds[3] = [0.135, 0.184]
	if prp == 2: prp_bounds[2] = [0.135, 0.184]
	if prp == 3: prp_bounds[4] = [0.135, 0.184]

	return prp_bounds



def analyze_with_slack( model_name, slack_val, eps=0.01, prp=1, verbose=0 ):

	#
	prp_bounds = generate_prp_bounds( prp )

	min_bound = np.array(prp_bounds)[:, 0]
	max_bound = np.array(prp_bounds)[:, 1]

	#
	policy_network = tf.keras.models.load_model( model_name, compile=False )
	
	#
	input_point = [ (a+b)/2 for [a, b] in prp_bounds ]

	#
	target = [1, 0, 0]

	#
	new_input_list = get_counter_example( policy_network, input_point, target, min_bound=min_bound, max_bound=max_bound, eps=eps )

	#
	max_diff_list = []
	for new_input in new_input_list:
		pre_activation_values = get_action(new_input, policy_network)[2]
		max_diff = 0
		if np.argmax(pre_activation_values) == 0:
			max_diff = max( np.abs(pre_activation_values - pre_activation_values[0]) )
		max_diff_list.append(max_diff)
	


	idx = np.argmax(max_diff_list)
	new_input = new_input_list[idx]
	max_diff = max_diff_list[idx]

	# 
	if verbose > 0:
		print( f"Testing model {model_name}:" )
		print( f"\tStarting from the input point: {input_point}" )
		print( f"\tWith output (softmax) is: {get_action(input_point, policy_network)[1]}" )
		print( f"\tAction before attack: {get_action(input_point, policy_network)[0]}")# (input {input_point})" ) 
		print( f"\tAction after attack: {get_action(new_input, policy_network)[0]}")# (input {new_input})" )
		print( f"\tThe founded input is: {new_input.round(4)[0]}" )
		print( f"\tThe founded output (softmax) is: {get_action(new_input, policy_network)[1]}" )
		print( f"\tThe founded output (pre-activation) is: {get_action(new_input, policy_network)[2]}" )
		print( f"\tThe maximum diference (pre-activation) is: {max_diff}" )

	# Return answer the question: is SAT with the given slack?
	# if True yes, if False unknwn
	return max_diff > slack_val



if __name__ == "__main__":

	# 
	#results_dict = {}

	#
	slack_list = [8.01]
	prp_list = [1, 2, 3]
	eps_list = [ 0.01, 0.001, 0.005 ]

	"""
	# Create the CSV file
	file_name = f"gradient_based_attack.csv"
	if os.path.exists(file_name): os.remove(file_name)
	csv_file = open(file_name, mode='x')
	writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n') #fieldnames=fieldnames, 
	writer.writeheader()
	"""

	#
	model_list = glob.glob( "adversarial_test/*/*.h5" )

	#
	print( f"\nStarting the analysis on #{len(model_list)} models" )


	#
	for eps in eps_list:

		#
		for prp in prp_list:

			#
			sat_for_slack = []

			#
			for slack in slack_list:

				number_of_sat = 0

				for model_name in model_list:

					#
					res = analyze_with_slack( model_name, slack, eps=eps, prp=prp, verbose=0 )
					#
					if res: number_of_sat += 1

				sat_for_slack.append( number_of_sat )

			
			print( f"\teps: {eps}, prp {prp} => {sat_for_slack}" )
			