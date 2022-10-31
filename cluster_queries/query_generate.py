import itertools


def create_string_baseline( 
	name, seed
):

	string = f"\
#!/bin/bash \n\n\
\
#SBATCH --job-name={name} \n\
#SBATCH --cpus-per-task=2 \n\
#SBATCH --mem-per-cpu=2G \n\
#SBATCH --output=/cs/labs/guykatz/dcorsi/output_files/{name}.output \n\
#SBATCH --mail-type=FAIL \n\
#SBATCH --mail-user=davide.corsi@mail.huji.ac.il \n\
#SBATCH --partition=long \n\
#SBATCH --time=24:0:0 \n\
#SBATCH --exclude=cb-[05-20],eye-[01-04],wadi-[01-05],gsm-[01-04],sm-[01-04],sm-[07-08],sm-[15-20],sulfur-[01-02],sulfur-[04-06],sulfur-[09-11] \n\n\
\
\
source /cs/labs/guykatz/dcorsi/venv/bin/activate \n\
cd /cs/labs/guykatz/dcorsi/python_training/ \n\
python training.py --run_name {name} --seed {seed} --verbose 2 --rules_active False --avoid_print True \
"
	return string


def create_string_rules( 

	name, 
	seed, 
	rule1_active, 
	rule2_active, 
	rule5_active,
	cost_limit=[],
	only_count_asif_penalty=True

):

	cost_limit_string = ''.join([f"{el} " for el in cost_limit])[:-1]

	string = f"\
#!/bin/bash \n\n\
\
#SBATCH --job-name={name} \n\
#SBATCH --cpus-per-task=2 \n\
#SBATCH --mem-per-cpu=2G \n\
#SBATCH --output=/cs/labs/guykatz/dcorsi/output_files/{name}.output \n\
#SBATCH --mail-type=FAIL \n\
#SBATCH --mail-user=hellow.world@fake.it \n\
#SBATCH --partition=long \n\
#SBATCH --time=24:0:0 \n\
#SBATCH --exclude=cb-[05-20],eye-[01-04],wadi-[01-05],gsm-[01-04],sm-[01-04],sm-[07-08],sm-[15-20],sulfur-[01-02],sulfur-[04-06],sulfur-[09-11] \n\n\
\
\
source /cs/labs/guykatz/dcorsi/venv/bin/activate \n\
cd /cs/labs/guykatz/dcorsi/python_training/ \n\
python training.py --run_name {name} --seed {seed} --cost_limit {cost_limit_string} --verbose 2 --avoid_print False --rules_list \
rule1_active={rule1_active} \
rule2_active={rule2_active} \
rule5_active={rule5_active} \
only_count_asif_penalty={only_count_asif_penalty} \
"
	return string




if __name__ == "__main__":

	parameters = [

		list(range( 0, 1)), 		#seeds
		[True],						#rule1_active
		[True],						#rule2_active
		[True],						#rule5_active
		[[1, 0, 5]],	#cost_limit

	]


	for combo in itertools.product( *parameters ):
		
		seed, rule1_active, rule2_active, rule5_active, cost_limit = combo

		cost_limit_string = ''.join([f"{el}-" for el in cost_limit])[:-1]
		name = "RUL" 
		name += f"_s{seed}_r1{rule1_active}_r2{rule2_active}_r5{rule5_active}_cl{cost_limit_string}"

		# Generate the query
		file_txt = create_string_rules( 

			name, 
			seed,
			rule1_active, 
			rule2_active, 
			rule5_active,
			cost_limit

		)

		text_file = open( f"launch_files/{name}.sbatch", "w" )
		n = text_file.write( file_txt )
		text_file.close()




"""
reward_penalty_std_dev = [0.2, 0.3, 0.4, 0.5]
reward_penalty = [1.0, 2, 3, 4, 5, 6, 7]
DO_NORMAL = [True]

rule1_active = [True]
rule2_active = [False] 
rule3_active = [False] 
rule5_active = [False]
"""