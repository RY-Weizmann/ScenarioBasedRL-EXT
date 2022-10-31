from env.robotic_navigation import RoboticNavigation
from env.robotic_navigation_es_extension import RoboticNavigation_ES_extension
from scripts.PPO import PPO
from scripts.MultiLagrangianPPO import MultiLagrangianPPO
import time, sys, argparse

class Training: 

	"""
	
		Main class of the training loop. 
		The trianing loop is performed with the 
		Proximal Policy Optimization algorithm (PPO) [1]

		[1] Proximal Policy Optimization Algorithms, 
			Schulman et al., 
			arXiv preprint arXiv:1707.06347, 2017

	"""

	# Default parameters
	verbose = 3
	episodes = 50000
	editor_run = False
	run_name = "STD"
	rebuild_limit = 5
	rules_active = True
	avoid_print = False

	# Default paramters for the paper 
	rules_list = ["rule1_active=True", "rule2_active=True", "rule5_active=True"]
	cost_limit = [0, 0, 0]

	#
	def __init__( self, **kwargs):

		# Override the default parameters with kwargs
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		#
		self.kwargs = kwargs

		# Fix Mandatory argument for RL algorithms
		self.kwargs['verbose'] = self.verbose
		self.kwargs['str_mod'] = self.run_name
		self.kwargs['avoid_print'] = self.avoid_print
		self.kwargs['cost_limit'] = self.cost_limit

		# Create the environment with the given params
		worker_id = int(round(time.time() % 1, 4)*1000)

		#
		if not self.rules_active:
			self.env = RoboticNavigation( editor_run=self.editor_run, worker_id=worker_id )	
			self.algo = PPO( self.env, **self.kwargs )
		
		#
		else:
			self.env = RoboticNavigation_ES_extension( editor_run=self.editor_run, worker_id=worker_id, rules_list=self.rules_list )
			self.algo = MultiLagrangianPPO( self.env, **self.kwargs )


	#
	def train( self ):

		#
		self.algo.loop( self.episodes )
		

# Call the main function
if __name__ == "__main__":

	# Launching without command line parameters (local run)
	if len(sys.argv) == 1:
		
		print( "Hello World Unity Simulator! \n")
		training = Training( editor_run=False, verbose=1, rules_active=True )
		training.train( )

	# Launching with command line parameters (cluster run)
	else:

		# Initialize the parser
		parser = argparse.ArgumentParser()

		# TRAINING DEFINITION
		parser.add_argument('--verbose', type=int )
		parser.add_argument('--episodes', type=int )
		parser.add_argument('--seed', type=int )
		parser.add_argument('--run_name', type=str )
		parser.add_argument('--cost_limit', nargs="+", type=int, default=[0, 0, 0] )
		parser.add_argument('--avoid_print', choices=('True','False') )
		parser.add_argument("--rules_list", nargs="+", default=["rule1_active=True", "rule2_active=True", "rule5_active=True"])

		# 
		arguments = parser.parse_args()
		arguments.avoid_print = (arguments.avoid_print == 'True')

		# Running the algorithm
		training = Training( **vars(arguments) )
		training.train()
