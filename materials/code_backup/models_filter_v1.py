from turtle import clear
import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys; sys.path.append("./")
from env.robotic_navigation import RoboticNavigation
import tensorflow as tf
import numpy as np
import glob, os, argparse, sys


def get_action( state, policy ):
	softmax_out = policy(state.reshape((1, -1))).numpy()
	selected_action = np.argmax( softmax_out )
	return selected_action


def main( env, policy_network, name, iterations=100 ):

	success = 0

	for _ in range(iterations):
		state = env.reset()

		while True:
			action = get_action( state, policy_network )
			state, _, done, info = env.step(action)
			if done: break
		if info["goal_reached"]: success += 1

	print( f"\t{name}: success: {success}" )
	return success


def clean_me( min_models, seeds_to_test ):

	pre_path = "/cs/labs/guykatz/dcorsi/robotic_training/log_to_test" # global path
	#pre_path = "./log" # local path

	try: 

		seed = np.random.randint(0, 1000)
		env = RoboticNavigation( worker_id=seed, random_seed=seed )
		print()

		for idx in seeds_to_test:

			for algo_name in ["REI", "PPO", "DQN"]: 
			
				experiment_dict = {}
				model_list = glob.glob(f"{pre_path}/{algo_name}*seed{idx}*/models/*.h5")
				print( f"Starting experiment on {algo_name} (seed {idx}) on #{len(model_list)} models" )
				print()

				if len( model_list ) <= min_models: continue

				for name in model_list:
					policy_network = tf.keras.models.load_model( name, compile=False )
					success = main( env, policy_network, name, iterations=10 )
					experiment_dict[name] = success		

					# Force output update (server only)
					sys.stdout.flush()		

				success_mean = np.mean( list(experiment_dict.values()) )
				for name, success in experiment_dict.items():
					model_list = glob.glob(f"{pre_path}/{algo_name}*seed{idx}*/models/*.h5")
					if len( model_list ) <= min_models: break
					if success < success_mean:
						print( f"\tdeleted: {name}")
						os.remove(name)		

	finally:
		print()
		env.close()
		

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--seed_to_test', type=int, required=True)
	seed = parser.parse_args().seed_to_test

	for _ in range(15): clean_me( min_models=5, seeds_to_test=[seed] )