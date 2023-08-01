# SMART HOME ENERGY MANAGEMENT
This repository storest the code for a conference paper about smart home energy management using reinforcement learning. It is possible for the user to train their own RL agent, as well as load pre-trained neural networks and test the agent's performance.

## INSTALLATION
To begin using the code, just clone the repository:
```git clone https://github.com/maticpokorn/smart_home_energy_management.git```

## USAGE
To use the functionalities of this repository, we must first import the module
```
import model
```
### Training an agent
The below code trains a new RL agent:
```
manager = model.SHEM()
manager.train()
```
The main class has a few superparameters:
- ```load```(default = False): tells the object to initialize from a pre-trained network
- ```path``` (default = None): path to directory from where to load if ```load = True```
- ```battery``` (default = 100): maximum battery capacity in kWh
- ```max_en``` (default = 5.8 / 4): maximum energy in kWh that can be charged or discharged from the battery in a single step (in our case in a 15 min window)
- ```n_days``` (default = 2): number of past days that will be stored in the environment state
#### Optional parameters for ```manager.train()```:
- ```a``` (default = 3): weight parameters for reward function
- ```b``` (default = 3): weight parameters for reward function
- ```n_episodes``` (default = 200): number of training episodes
- ```epsilon_reduce``` (default = 0.98): factor by which the exploration rate (epsilon) is reduced after each episode
- ```n_days``` (default = 2): number of past days that will be stored in the environment state
- ```n_steps``` (dafault = 7 * 24 * 4): number of steps to take each episode (the default is equal to one week)

### Saving
The following code saves the manager to a directory called ```'my_net'```:
```
manager.save('my_net')
```

### Loading from a saved directory
To load a pre-trained model, we must create a new manager:
```
manager = model.SHEM(load=True, path='saved_nets/my_net')
```

### Testing
```
manager.test()
```
Due to the unstable nature of the DQN algorithm, meaningful trainig results can sometimes only be achieved by significantly increasing the number of episodes or training the agent over and over (>100 times) until results are satisfactory. This is very computationally intensive and is also the reason why pre-trained managers are important for demonstrating the performance of the algorithm.

## FOLDER STRUCTURE
```
project
│   README.md
│   adjust_dataset.py
|   dqn.py
|   env.py
|   model.py
|   reshape_ev_charging.py
│
└───data
│   │   rtp.csv
│   │   tou.csv
│   |   tou2.csv
│
└───saved_nets
|   │...
│
└───raw_data
    │...    
```
