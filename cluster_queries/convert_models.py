from hashlib import new
import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys; sys.path.append("./")
import tensorflow as tf
import glob

#"../materials/best_models/DDQN_id141_ep12681.h5"

def rename(model, layer, new_name):
    def _get_node_suffix(name):
        for old_name in old_nodes:
            if old_name.startswith(name):
                return old_name[len(name):]

    old_name = layer.name
    old_nodes = list(model._network_nodes)
    new_nodes = []

    for l in model.layers:
        if l.name == old_name:
            l._name = new_name
            new_nodes.append(new_name + _get_node_suffix(old_name))
        else:
            new_nodes.append(l.name + _get_node_suffix(l.name))
    model._network_nodes = set(new_nodes)

#
original_folder_name = "python_training/rule_all_models"
model_list = glob.glob( f"{original_folder_name}/*.h5" )

#
for name in model_list:

	policy_network = tf.keras.models.load_model( name, compile=False )

	policy_network._name = "robotic_model"
	rename(policy_network, policy_network.layers[-2], "last_layer")
	rename(policy_network, policy_network.layers[-1], "output_layer")

	folder_name = name.split("/")[-1][:-3].split("_")[1]
	folder_name += "_" + name.split("/")[-1][:-3].split("_")[-2]
	folder_name = f"rule_all_{folder_name}"

	os.mkdir( f"{original_folder_name}/converted/{folder_name}" )
	policy_network.save( f"{original_folder_name}/converted/{folder_name}" )

