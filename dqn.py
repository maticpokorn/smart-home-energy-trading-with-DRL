import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import json

BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON_INIT = 0.99
EPSILON = EPSILON_INIT  # 0.99            # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target DQN update frequency
MEMORY_CAPACITY = 2000  # size of the replay buffer

ENV_A_SHAPE = (2,)
#ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# class which defines the structure of the neural network of the DQN. Note that the input is the state, and the output a set of Q values, one for each action
class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 8)  # linear input layer with N_STATES number of nodes
        self.fc1.weight.data.normal_(0, 0.1)  # initialization of weights of this layer
        self.fc2 = nn.Linear(8, 8)  # hidden layer with 8 nodes
        self.fc2.weight.data.normal_(0, 0.1)  # initialization of weights of this layer
        self.out = nn.Linear(8, n_actions)  # output layer with N_ACTIONS number of nodes
        self.out.weight.data.normal_(0, 0.1)  # initialization of weights of this layer

    def forward(self, x):  # pass state into the NN and retrieve the output
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


# the DQN class, which incorporates the neural network above, and includes the replay buffer and the training process
class DQN(object):
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

        self.eval_net, self.target_net = Net(self.n_states, self.n_actions).to(device), Net(self.n_states, self.n_actions).to(device)

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, self.n_states * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss().to(device)

    def choose_action(self, x, epsilon):
        x = torch.FloatTensor(x).to(device)
        x = torch.unsqueeze(x, 0)
        # input only one sample
        if np.random.uniform() > epsilon:  # greedy action, for exploitation
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.detach().cpu().numpy()
            action = action if isinstance(action, int) else action[0]
        else:  # random action, for exploration
            action = np.random.randint(0, self.n_actions)
            action = action if isinstance(action, int) else action[0]
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states]).to(device)
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:]).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
    def save(self, path):
        torch.save(self.eval_net, path + '/eval_net.pt')
        torch.save(self.target_net, path + '/target_net.pt')
        d = {
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learn_step_counter': self.learn_step_counter,
            'memory_counter': self.memory_counter,
            'memory': self.memory.tolist()
            }
        with open(path + "/params.json", "w") as outfile:
            json.dump(d, outfile)
        
        
    def load(self, path):
        
        d = {}
        
        with open(path + "/params.json", "r") as openfile:
            d = json.load(openfile)
            
        self.n_states = d['n_states']
        self.n_actions = d['n_actions']
        self.learn_step_counter = d['learn_step_counter']
        self.memory_counter = d['memory_counter']
        self.memory = np.array(d['memory'])
        
        self.eval_net = torch.load(path + '/eval_net.pt', map_location=torch.device('cpu')).to(device)
        self.target_net = torch.load(path + '/target_net.pt', map_location=torch.device('cpu')).to(device)
        self.eval_net.eval()
        self.target_net.eval()