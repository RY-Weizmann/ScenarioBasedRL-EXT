import csv, glob, shutil

data = [
		glob.glob("AAA.Data/Data_Batch_*/robotic_training/log/*PPO*"),
		glob.glob("AAA.Data/Data_Batch_*/robotic_training/log/*DQN*"),
		glob.glob("AAA.Data/Data_Batch_*/robotic_training/log/*REI*")
]

for PPO_folder in data[0]:
	folder_name = PPO_folder.split("/")[-1]
	print(folder_name)
	shutil.copy( f"{PPO_folder}/run_stats.csv", f"SAVED/{folder_name}.csv" )


for DQN_folder in data[1]:
	folder_name = DQN_folder.split("/")[-1]
	print(folder_name)
	shutil.copy( f"{DQN_folder}/run_stats.csv", f"SAVED/{folder_name}.csv" )


for REI_folder in data[2]:
	folder_name = REI_folder.split("/")[-1]
	print(folder_name)
	shutil.copy( f"{REI_folder}/run_stats.csv", f"SAVED/{folder_name}.csv" )
