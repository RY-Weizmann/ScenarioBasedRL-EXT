#!/bin/bash


#SBATCH --job-name=killable_/cs/labs/guykatz/guyam/rl_ensembles/ipqs/good_and_bad_pair/policy_experiments_deep_buffer/short_episodes_during_training/pair_906_929/ipq_files/pair_906_929_inDistribution_srToBwRatioUb_1.5_slack_1_part_a
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cs/labs/guykatz/guyam/rl_ensembles/ipqs/good_and_bad_pair/policy_experiments_deep_buffer/short_episodes_during_training/pair_906_929/output_files/pair_906_929_inDistribution_srToBwRatioUb_1.5_slack_1_part_a.output
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=guy.amir2@mail.huji.ac.il
#SBATCH --partition=long
#SBATCH --time=12:0:0

/cs/labs/guykatz/guyam/rl_ensembles/Marabou/build/Marabou --input-query=/cs/labs/guykatz/guyam/rl_ensembles/ipqs/good_and_bad_pair/policy_experiments_deep_buffer/short_episodes_during_training/pair_906_929/ipq_files/pair_906_929_inDistribution_srToBwRatioUb_1.5_slack_1_part_a