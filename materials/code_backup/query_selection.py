import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys; sys.path.append("./")
from env.robotic_navigation import RoboticNavigation
import tensorflow as tf
import numpy as np
import glob, os, argparse, sys, shutil


def get_action( state, policy ):
	softmax_out = policy(state.reshape((1, -1))).numpy()
	selected_action = np.argmax( softmax_out )
	return selected_action


def run( seed, policy_network, iterations=100 ):

	env = RoboticNavigation( worker_id=seed, random_seed=seed )
	
	success = 0
	cost = 0

	for _ in range(iterations):
		state = env.reset()

		while True:
			action = get_action( state, policy_network )
			state, _, done, info = env.step(action)
			cost += info["cost"]
			if done: break

		if info["goal_reached"]: success += 1

	env.close()

	return success, int(cost/iterations)


def main():
	
	env_seed = np.random.randint(0, 1000)
	

	import random
	
	models_to_test_0 = glob.glob("log/DEF0.0_*/models/*.h5")
	models_to_test_1 = glob.glob("log/DEF0.01_*/models/*.h5")
	models_to_test_2 = glob.glob("log/DEF0.02_*/models/*.h5")
	models_to_test_3 = glob.glob("log/DEF0.03_*/models/*.h5")

	models_to_test_0 = random.sample(models_to_test_0, 10)
	models_to_test_1 = random.sample(models_to_test_1, 10)
	models_to_test_2 = random.sample(models_to_test_2, 10)
	models_to_test_3 = random.sample(models_to_test_3, 10)

	models_to_test = models_to_test_0 + models_to_test_1 + models_to_test_2 + models_to_test_3
	
	print( f"\n\nStarting the analysis on {len(models_to_test)} models... ")

	for idx, model_name in enumerate(models_to_test):

		policy_network = tf.keras.models.load_model( model_name, compile=False )
		success, cost = run( env_seed, policy_network, iterations=100 )

		model_code = f"{model_name.split('/')[-1][:-3]}_success{success}_cost{cost}.h5"
		print( f"\t[{idx}/{len(models_to_test)}] {model_code}" )

		shutil.copy( f"{model_name}", f"log/processed_models/{model_code}" )


if __name__ == "__main__":
	main()