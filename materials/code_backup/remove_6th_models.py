model_list = []
model_list += glob.glob( "adversarial_test/models_1_2/*h5" )
model_list += glob.glob( "adversarial_test/models_3/*h5" )

print(len(model_list))

seeds = []
for model in model_list: seeds.append( int(model.split("_")[-2][2:]) )
seeds = list(set(seeds))
seeds.sort()

for seed in seeds:
	print(seed)

	DQN_MODELS = glob.glob("Best_Models_h5/DQN/*.h5")
	REI_MODELS = glob.glob("Best_Models_h5/REI/*.h5")
	PPO_MODELS = glob.glob("Best_Models_h5/PPO/*.h5")

	print( len(DQN_MODELS) )
	print( len(REI_MODELS) )
	print( len(PPO_MODELS) )
	print()


	model_list = REI_MODELS
	seeds = []
	for model in model_list: seeds.append( int(model.split("_")[-2][2:]) )
	seeds = list(set(seeds))
	seeds.sort()
	
	
	for seed in seeds:
		element_with_seed = glob.glob(f"Best_Models_h5/REI/*id{seed}_*.h5")
		#print(len(element_with_seed))
		if len(element_with_seed) > 5: 
			os.remove(element_with_seed[0])
	

	
	import shutil
	for rei_model in glob.glob("log_*/*REI*/models/*.h5"):
		model_name = rei_model.split("/")[-1]
		shutil.copy( rei_model, f"Best_Models_h5/REI/{model_name}" )
	


	quit()