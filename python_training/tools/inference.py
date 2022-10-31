import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys; sys.path.append("./")
from env.robotic_navigation import RoboticNavigation
import tensorflow as tf
import numpy as np
import time


def get_action( state, policy ):
	
	#
	softmax_out = policy(state.reshape((1, -1)))
	selected_action = np.random.choice(3, p=softmax_out.numpy()[0])

	#
	deterministic_output = softmax_out.numpy()[0]
	deterministic_action = np.argmax(deterministic_output)
	
	#
	return selected_action


def main( env, policy_network, iterations=30 ):

	for ep in range(iterations):

		state = env.reset()

		while True:
			action = get_action( state, policy_network )		
			state, _, done, info = env.step(action)			
			if done: break



if __name__ == "__main__":
	
	model_path = "tools/models/RUL_s221_r1True_r2True_r5True_cl1-0-5_id221_ep46023_srate89.h5"
	policy_network = tf.keras.models.load_model( model_path )
	
	try:
		env = RoboticNavigation( editor_run=True, random_seed=0 )
		success = main( env, policy_network, iterations=30 )
		print( success )

	finally:
		env.close()