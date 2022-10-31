from reinforcement_plotter import ReinforcementPlotter
import glob, numpy, os, shutil, csv
import pandas as pd


"""
data = [
		glob.glob("log/*/run_stats.csv")
]

limit = 50000
for data_array in data:
	for file_name in data_array:
		with open(file_name) as f:
			file_size = sum(1 for line in f) - 1
			if file_size < limit: 
				folder_name = file_name.split("/")[1]
				name = "log/"+folder_name
				print( f"\tdeleted: {name}")
				shutil.rmtree(name, ignore_errors=True)	
quit()
"""


data = [
	#glob.glob("../materials/models_total_batch_1/*r1False_r2False_r5False*/run_stats.csv"),

	#glob.glob("../materials/models_total_batch_1/*r1True_r2False_r5False*/run_stats.csv")	
	glob.glob("../materials/models_total_batch_2/*/run_stats.csv"),


	#glob.glob("../materials/models_total_batch_1/*r1True_r2True_r5True*/run_stats.csv")
	glob.glob("log/*_1.5_*/run_stats.csv")
	#[
	#"../materials/models_total_batch_1/RUL_s176_r1False_r2False_r5False_cl1_20220603_215618/run_stats.csv",
	#"../materials/models_total_batch_1/RUL_s112_r1False_r2False_r5False_cl1_20220603_192623/run_stats.csv",
	#],


	#[
	#"../materials/models_total_batch_1/RUL_s143_r1False_r2False_r5False_cl1_20220603_192635/run_stats.csv",
	#"../materials/models_total_batch_1/RUL_s112_r1False_r2False_r5False_cl1_20220603_192623/run_stats.csv",
	#],	

	#[
	#"../materials/models_total_batch_2/RUL_s80_r1True_r2True_r5True_cl1-0-5_20220606_134721/run_stats.csv",
	#"../materials/models_total_batch_2/RUL_s211_r1True_r2True_r5True_cl1-1-10_20220606_140525/run_stats.csv",
	#]
]


# Plot The Results
for key in [
	"success",
	#"back&forth",
	#"sum_of_back_and_forth",
	#"sum_of_6_or_more_same_direction_turns", 
	#"sum_of_fwd_encouragement"
]:

	plotter = ReinforcementPlotter( x_label="episode", y_label="success rate", title=f"" )
	plotter.load_array( data, key=key, ref_line=None ) # success, changes, cost, step
	plotter.process_data( rolling_window=1000 )

	#
	plotter.render_std( 

		#labels=["Baseline", "Rule 1"],#, "All Rules"], 
		labels=["Optmized Lag.", "Fixed Pen. (1.0)"],
		colors=["r", "k"],
		styles=["-", "--", ":", "."],
		save_image=True 

	)

	break
