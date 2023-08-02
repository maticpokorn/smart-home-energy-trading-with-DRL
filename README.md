# SMART HOME ENERGY MANAGEMENT WITH DRL
This python library storest code for a conference paper about smart home energy management using Deep Reinforcement Learning. It is possible for the user to train their own DRL agent, as well as load pre-trained models and test the agent's performance.

## INSTALLATION
To begin using the code, just clone the repository:
```git clone https://github.com/maticpokorn/HEMS_DQN.git```

## USAGE
To use the functionalities of this repository, we must first import the module into our own ```.py``` or ```.ipynb``` file:
```
import HEMS
```
### Training an agent
The below code trains a new RL agent:
```
manager = HEMS.HEMS()
manager.train()
```
The ```HEMS``` class has a few parameters:
- ```load```(default = False): tells the object to initialize from a pre-trained network
- ```path``` (default = None): path to directory from where to load if ```load = True```
- ```battery``` (default = 20): maximum battery capacity in kWh
- ```max_en``` (default = 1.5): maximum energy in kWh that can be charged or discharged from the battery in a single step (in our case in a 15 min window)
- ```eff``` (default = 0.9): input / output battery efficiency
- ```price_coefs``` (default = [2,1]): price coefficients. They specifiy by how much the market price from the dataset should be multiplied to get buying and selling price of energy. This is an artificial way to introduce Distribution System Operator charges.
- ```n_days``` (default = 2): number of past days that will be stored in the environment state
- ```data_path``` (default = 'data/rtp.csv'): specifies the dataset with which pricing model to use
#### Optional parameters for ```manager.train()``` function:
- ```a``` (default = 3): weight parameter for reward function
- ```b``` (default = 3): weight parameter for reward function
- ```n_episodes``` (default = 200): number of training episodes
- ```epsilon_reduce``` (default = 0.98): factor by which the exploration rate (epsilon) is reduced after each episode
- ```n_days``` (default = 2): number of past days that will be stored in the environment state
- ```n_steps``` (dafault = 7 * 24 * 4): number of steps to take each episode (the default is equal to one week)

### Saving
The following code saves the manager to a directory called ```'my_net'```:
```
manager.save('saved_nets/my_net')
```

### Loading from a saved directory
To load a pre-trained model, we must create a new manager with the ```load``` parameter set to ```True```:
```
manager = HEMS.HEMS(load=True, path='saved_nets/my_net')
```

### Testing
```
manager.test()
```
Due to the unstable nature of the DQN algorithm, meaningful trainig results can sometimes only be achieved by significantly increasing the number of episodes or training the agent over and over (>10 times) until results are satisfactory. This is very computationally intensive and is also the reason why pre-trained managers are important for demonstrating the performance of the algorithm.

## WHAT HAPPENS UNDER THE HOOD
This is a Deep Reinforcement Learning library for Smart Home Energy Management. In RL, we usually need to define a RL **ENVIRONMENT** on which a RL **AGENT** is trained. That is why the ```HEMS``` class initialises an ```Env``` class from ```env.py``` (the environment) and a ```DQN``` class from ```dqn.py``` (the agent - in this case the agent is defined as an algorithm called DQN). If ```load == False``` the environment and agent are initilaised inside the ```train()``` function. If ```load == True``` the environment and agent are loaded from the saved folder.

## FOLDER STRUCTURE
```
project
│   README.md
|   dqn.py
|   env.py
|   HEMS.py
│
└───data
│   │   rtp.csv
│   │   tou.csv
│   |   tou2.csv
│
└───saved_nets
    │...  
```
