import glob

complete_list = glob.glob("AAA.Data/SafeRoboticsPaper_Data/Data_Batch*/robotic_training/log/*")
print(len(complete_list))


seed_list = []
for current_training in complete_list:
	seed = current_training.split("seed")[-1].split("_")[0]
	seed_list.append(int(seed))


seed_ordered = list(set(seed_list))
seed_ordered.sort()

print(seed_ordered)
print(len(seed_ordered))