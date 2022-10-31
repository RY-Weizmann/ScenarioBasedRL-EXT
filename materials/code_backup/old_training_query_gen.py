import itertools

def create_string( 
	job_name, 
	seed, 
	run_name, 
	episodes=4000,  
	verbose=3, 
	critic_epoch=30, 
	trajectory_update=10, 
	nodes=32,
	batch_size=128,
	reward_penalty=0
):

	string = f"\
#!/bin/bash \n\n\
\
#SBATCH --job-name={job_name} \n\
#SBATCH --cpus-per-task=2 \n\
#SBATCH --mem-per-cpu=2G \n\
#SBATCH --output=/cs/labs/guykatz/dcorsi/output_files/{job_name}.output \n\
#SBATCH --mail-type=FAIL \n\
#SBATCH --mail-user=davide.corsi@mail.huji.ac.il \n\
#SBATCH --partition=long \n\
#SBATCH --time=24:0:0 \n\
#SBATCH --exclude=cb-[05-20],eye-[01-04],wadi-[01-05],gsm-[01-04],sm-[01-04],sm-[07-08],sm-[15-20],sulfur-[01-02],sulfur-[04-06],sulfur-[09-11] \n\n\
\
\
source /cs/labs/guykatz/dcorsi/venv/bin/activate \n\
cd /cs/labs/guykatz/dcorsi/python_training/ \n\
python training.py --run_name {run_name} --verbose {verbose} \
--episodes {episodes} --seed {seed} --nodes {nodes} \
--trajectory_update {trajectory_update} --critic_epoch {critic_epoch} \
--batch_size {batch_size} --reward_penalty {reward_penalty} \
"
	return string


if __name__ == "__main__":

	parameters = [

		list(range( 0, 10)), #seeds
		[0.0, 0.03] #reward_penalty

	]

	for combo in itertools.product( *parameters ):
		
		seed, reward_penalty = combo

		run_name = f"s{seed}_p{reward_penalty}_"
		job_name = f"{run_name}"
		
		# Generate the query
		file_txt = create_string( 

			job_name, 
			seed, 
			run_name,
			episodes=4000,
			reward_penalty=reward_penalty

		)

		text_file = open( f"launch_files/{job_name}.sbatch", "w" )
		n = text_file.write( file_txt )
		text_file.close()


if __name__ == "__main__":


	if len(sys.argv) == 1:
		
		print( "Hello World Unity Simulator! \n")
		for seed in [10, 20, 30]:
			#training = Training( editor_run=False, verbose=1, rules_active=True  )
			training = Training( run_name="DEF", verbose=2, seed=seed, rules_active=False )
			#training = Training( run_name="PRP", verbose=2, seed=seed, rules_active=True )
			training.train( )

	else:

		# initialize the parser
		parser = argparse.ArgumentParser()

		# TRAINING DEFINITION
		parser.add_argument('--verbose', type=int, required=True)
		parser.add_argument('--episodes', type=int, required=True)
		parser.add_argument('--seed', type=int, required=False)
		parser.add_argument('--run_name', type=str, required=False)

		
		# Training Parameters
		parser.add_argument('--nodes', type=int, required=False)
		parser.add_argument('--batch_size', type=int, required=False)
		parser.add_argument('--critic_epoch', type=int, required=False)
		parser.add_argument('--trajectory_update', type=int, required=False)

		# rULES Parameters
		parser.add_argument('--reward_penalty', type=float, required=False)

		# Running the algorithm
		training = Training( **vars(parser.parse_args()) )
		training.train()