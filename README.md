### Disclaimer

*NB: work in progress repo, remember to fix typos and content when have time...*

# ScenarioBasedRL
All the following scripts must be run from the directory **python_training**

## Quick Start: Training
To run the training, use the script *training.py*. 
```
python training.py
```

The script will run with the default parameters. All the paramters can be changed in line 77, in the following example the hyperparamters *editor_run*, *verbose* and *rules_active* have been changed:
```
training = Training( editor_run=False, verbose=1, rules_active=True )
```

Following a list of the most important hyperparameters for the training:
- **editor_run**: if False the simulation engine will be the built version of the environment, if True it run inside the Unity3D engine. [default=False]. 
- **verbose**: if > 1 the algorithm saves the results in a .csv file, if > 2 the algorithm saves also the trained DNN models (the saved models can be used in inference). [default=2]
- **rules_active**: if False the algorithm run with the standard PPO (without the injection of the rules), if True the algorithm computes the number of violations as a cost function, running Lagrangian PPO as training algorithm. [default=True]
- **cost_limit**: limit for the violations allowed of each rule, requires an array of integer that represent the threshold for each cost function (or violations). [default=[1, 0, 5]]

## Quick Start: Inference

To run the inference, use the script *inference.py*.
```
python tools/inference.py
```

The script will run with the default parameters. In the main file it is possible to change all the basic functions:
```
env = RoboticNavigation( editor_run=True, random_seed=9 )
success = main( env, policy_network, iterations=30 )
```

Following a list of the most important hyperparameters for the inference:
- **editor_run**: if False the simulation engine will be the built version of the environment, if True it run inside the Unity3D engine. [default=False]. 
- **random_seed**: random seed for reproducibility, this value control the spwaining sequence of the agent in the environment. The same seed generates the same sequence of spwaining point (and necessary trajectory to solve). [default=0]
- **policy_network**: the path to the TF2 neural network, the models are typically saved in the training phase. Two working somple can be found in the folder *tools/models*.
- **iterations**: the number of episode to run before returning the results and close the environment. [default=30]


## Project Structure:
- **materials**: Folder for all the other materials *(saved models, images, video, ...)*
- **python_training**: Folder holding the Python code for the DRL agent controlling the robot
- **unity_project**: Folder holding the Unity Robotic Simulation project

NB: all the python files **query_*.py** are for the cluster runs, ignore if not needed.*


## Output Description:
The output (or **action space**) of the network is of 3 elements:

- **Action 0 (forward):** move forward of a step of **0.05** units
- **Action 1 (turn right):** turn right with an angle of **30deg**
- **Action 2 (turn left):** turn left with an angle of **30deg**


## Input Description:
The input (or **state**) of the neural netowrk is an array of 9 elements:
- **Input 0, 1, 2, 3, 4, 5 and 6**: the input of the lidar scan, from left to right. The returned value is normalized between 0 and 1, where 0 is the minimum distance from an obstacle and >= 1 means 'free direction'.
- **Input 7**: is the heading, i.e., the angle between the agent and the target.
- **Input 8**: is the distance between the agent and the target, normalized between 0 and 1 (where 1 is the size of the full map). Input 7 and 8 togheter generate the polar coordinates of the relative position of the target respect to the agent.

<br/><br/>
<img src="materials/images/state_description.png" align="middle" width="500"/>

*Fig. description of the inputs 0, ..., 6 (lidar scan) and input 7 goal angle*
<br/><br/>

### Considerations about the values:
- A single step forward (action 0) towards an obstacle, reduce the lidar scan of a value of **0.05**.
- If the value of the central lidar is lower than **0.185** and the agent select the action *forward*, it will collide for sure (in at most 3 steps), without any chance to avoid the impact.
- The agent touch an obstacle when the lidar value is less-equal than **0.08**.
- The agent reaches the goal if the distance (input 8) is lower than 0.1, however for a target closer than **0.15** reaching the goal is a trivial action.

### Consideration about the trivial bounds:
Even if the normalized values are in the interval [0, 1], given the size of the robot, an obstacle can not be closer than a certain value (otherwise is already a collision), for a similar reason the value of the distance from the target can not be lower than a value (otherwise is already reached). Summarizing, the *corrected trivial bounds* are:

- **Input 0, 1, 2, 3, 4, 5 and 6**: [0.2, 1]
- **Input 7**: [0, 1]
- **Input 8**: [0.15, 1]


## Notes on the Python Environment and Unity Version:
The following packages must be installed via **pip**:
```
- gym==0.20.0
- gym-unity==0.28.0
- mlagents-envs==0.28.0
- tensorflow==2.8.0
```

Tested version for **Unity** and **ML-Agents**:
```
- Unity Hub v3.1.2
- Unity v2021.3
- ML-Agents v2.3.0
```


## Additional Notes

### Scenario Based Programming Rules:
Important note: on 25-5-2022 the implementation of rule5 was changed: instead of encouraging (i.e., giving positive reward) to move
fwd when target is ahead, it is now penalizing (i.e., negative reward) for when the target is ahead, sensors are clear BUT the agent command is NOT FWD.
Reporting stay the same, although the reversed loogic (counting sum_of_fwd_encouragement, althoug this is now counting penalties)


### Execution Tips:
- Use conda env: 'conda activate RoboLearn'
- Can use many flags in the command-line:
  : --verbose <level 0-3>, not realy sure how each is defined...
  : OBSOLETE, being calculated based on any specific rule being active// --rules_active <True, False> controls what is says..
  : --avoid_print <True, False> reduce noise level, esp for cluster runs
  : --rules_list
    :: Available here:
     reward_penalty=4.5 <Float, penalty mean>
     reward_penalty_std_dev=0.5 <Float, penalty mean deviation>
     DO_NORMAL=True <True, False> Can have EXCULISVE EITHER of: DO_FIXED or DO_NORMAL or DO_UNIFORM
     rule1_active=True # scenario 1: do not go back-and-forth, e.g, right->left / left->right / right->left
     rule2_active=True # scenario 2: do not go in circle, e.g., more then 6 consecutive right or left turns
     rule3_active=True #do  not go in long circles, e.g., get back to where the same target direction and distance have already  seen before
     rule4_active=True #PERMANENTLY DISSABLED  distance to target must decrease over time!
     rule5_active=True #penalize if the target is straight ahead and the agent does not move FWD!
     only_count_asif_penalty=True <True, False> When True, the reward is NOT modified. Rule violations are reported.

Example:
python.exe training.py --verbose 3 --avoid_print True --rules_list reward_penalty_std_dev=0.5 reward_penalty=4.5 DO_NORMAL=True rule1_active=True

python.exe training.py --verbose 3 --avoid_print True --rules_list reward_penalty_std_dev=0.5 reward_penalty=4.5 DO_NORMAL=True rule1_active=True rule2_active=True rule3_active=True rule5_active=True only_count_asif_penalty=True

* The "--rules_list" must be LAST and contains paramteres that are passed to the rules system.
