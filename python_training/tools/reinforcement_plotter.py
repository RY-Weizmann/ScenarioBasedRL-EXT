import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import deque
import warnings; warnings.filterwarnings("ignore")

class ReinforcementPlotter():

	def __init__( self, x_label="X Label", y_label="Y Label", title="", cap=None ):

		self.fig, self.ax = plt.subplots(1)
		self.ax.spines["top"].set_visible(False)    
		self.ax.spines["bottom"].set_visible(False)    
		self.ax.spines["right"].set_visible(False)    
		self.ax.spines["left"].set_visible(False)  
		self.ax.set_facecolor('#eaeaf2')
		plt.grid(color='#ffffff', linestyle='-', linewidth=1)
		plt.xticks(fontsize=18)
		plt.yticks(fontsize=18)

		plt.xlabel(x_label, fontsize=18)
		plt.ylabel(y_label, fontsize=18)
		plt.title(title, fontsize=18)

		fig = plt.gcf()
		fig.set_size_inches(9, 5)

		self.data_arrays = []
		self.array_len = -1

		self.mean_array = []
		self.var_array =  []
		self.max_array = []
		self.min_array = []

		self.cap = cap
		self.run_name = f"{title}_{y_label}"

		self.key = None

	
	def load_array( self, file_name_arrays, key="reward", ref_line=None, early_stop=None ):

		self.key = key
		data_arrays = []

		for array_set in file_name_arrays:
			data_arrays.append([])
			for name in array_set:
				data_arrays[-1].append([])
				with open(name) as csv_file:
					csv_reader = csv.reader(csv_file, delimiter=',')
					for line, row in enumerate(csv_reader):
						if line == 0: 
							try:
								key_index = row.index(key) 
							except Exception: 
								print( f"Requested key ({key}) not in the CSV! options: {row}")
								quit()
							continue
						data_arrays[-1][-1].append( float(row[key_index]) )
				data_arrays[-1][-1] = np.array(data_arrays[-1][-1])

		if(early_stop == None): self.array_len = min([min([len(el) for el in array_set]) for array_set in data_arrays])
		else: self.array_len = early_stop
		self.data_arrays = np.array([[el[:self.array_len] for el in array_set] for array_set in data_arrays], dtype=object)

		if ref_line is not None: plt.axhline(y=ref_line, color='k', linestyle='--', linewidth=1.5 )

	
	def render_std( self, labels, colors, styles=None, save_image=False ):
		err_msg = "load some data before the render!"
		assert self.array_len > 0, err_msg

		if styles is None: styles = ['-' for _ in range(len(labels))]

		for mean_values, var_values, mins, maxis, label, color, style in zip(self.mean_array, self.var_array, self.min_array, self.max_array, labels, colors, styles):
			self.ax.plot(self.x_axes, mean_values, label=label, color=color, linestyle=style, linewidth=1.5 )
			self.ax.fill_between(self.x_axes, mean_values+var_values, mean_values-var_values, facecolor=color, alpha=0.3)
			#self.ax.fill_between(self.x_axes, maxis, mins, facecolor=color, alpha=0.3)


		if self.key == "success": 
			plt.ylim( bottom=0, top=1)

		elif self.key == "back&forth" or self.key == "sum_of_fwd_encouragement": 
			plt.ylim( bottom=0, top=30)

		elif self.key == "sum_of_6_or_more_same_direction_turns": 
			plt.ylim( bottom=0, top=0.09)
			labels = [str(i) for i in range(10)] 
			self.ax.set_yticklabels(labels)

		else:
			raise ValueError
		
		#self.ax.legend(loc='best', fontsize=18)
		#self.ax.legend(loc='lower left', fontsize=18)
		self.ax.legend(loc='upper left', fontsize=18)

		plt.subplots_adjust( bottom=0.14, left=0.085, right=0.98, top=0.97, wspace=0.2, hspace=0.2 )
		
		if not save_image: plt.show()
		else: plt.savefig(f"{self.run_name}.pdf")


	def process_data( self, rolling_window=1, starting_pointer=0 ):		
		rolling_queue = deque(maxlen=rolling_window)
		self.x_axes = [i for i in range(self.array_len-starting_pointer)]

		for array_set in self.data_arrays:
			for array in array_set:
				for i in range(self.array_len):
					rolling_queue.append(array[i])
					array[i] = np.mean(rolling_queue)
					if self.cap != None: array[i] = min(array[i], self.cap)
				rolling_queue.clear()

		# Fix for different array size
		self.data_arrays = np.array([np.array(el) for el in self.data_arrays], dtype=object)
		
		self.mean_array = np.array([[np.mean(array_set[:, i]) for i in range(self.array_len)][starting_pointer:] for array_set in self.data_arrays])
		self.var_array =  np.array([[np.std(array_set[:, i]) for i in range(self.array_len)][starting_pointer:] for array_set in self.data_arrays])
		self.max_array = [[np.max(array_set[:, i]) for i in range(self.array_len)][starting_pointer:] for array_set in self.data_arrays]
		self.min_array = [[np.min(array_set[:, i]) for i in range(self.array_len)][starting_pointer:] for array_set in self.data_arrays]
				