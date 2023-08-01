import pandas as pd
import numpy as np
import env
import dqn
from tqdm import tqdm
import shutil
import os
import json
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# HEMS = Home Energy Management System
class HEMS:
    
    def __init__(self, load=False, path=None, battery=20, max_en=6/4, eff=0.9, price_coefs=[2,1], n_days=2, data_path='data/rtp.csv'):
        self.memory_capacity = 2000
        self.agent = None
        self.path = path
        if load:
            self.load_set_attributes(path)
        else:    
            self.battery = battery
            self.max_en = max_en
            self.eff = eff
            self.price_coefs = price_coefs
            self.df = pd.read_csv(data_path)        
            self.epsilon = 1
            self.n_days = 2
        #print(self.battery, self.max_en, self.eff, self.price_coefs, self.epsilon)
        print(f'YOU HAVE INITIALISED YOUR HEMS (Home Energy Management System) WITH FOLLOWING SPECIFICATIONS:\nBattery size: {self.battery} kWh\nMax input/output energy per step: {self.max_en} kWh\nBattery efficiency: {int(self.eff * 100)}%\nEnergy selling price is set to {self.price_coefs[1]} * dataset market price\nEnergy buying price is set to {self.price_coefs[0]} * dataset market price\nDataset was loaded from: {path if load else data_path}')
        
    def train(self, a=3, b=3, n_episodes=200, epsilon_reduce=0.98, n_days=2, n_steps=7*24*4):
        df = self.df
        seed = 0
        # ENVIRONMENT AND AGENT INITIALIZATION
        envRL = env.Env(df, self.battery, self.max_en, self.eff, self.price_coefs, n_days, n_steps)
        envRL.reset(seed)
    
        self.agent = dqn.DQN(envRL.next_observation_normalized().shape[0], 4)

        # TRAINING
        print('Trainig in progress...')
        epsilon = self.epsilon
        for episode in tqdm(range(n_episodes)):
            epsilon = epsilon * epsilon_reduce
            cost_dqn = self.run_train_episode(n_steps, envRL, self.agent, epsilon, a, b)
            
        self.epsilon = epsilon
    
    def save(self, name):
        path = name
        if os.path.exists(path):
            print("Path already exists, please check and remove files at this location or rename your model")
        else:
            os.mkdir(path)       
            self.agent.save(path)
            self.save_properties(path, self.df, self.n_days, self.epsilon, 3, 3)
            print("model saved to location: " + path)
    
    def test(self, a=3, b=3, start=30000, steps=None):
        print("--- TESTING ---")
        df = self.df
        epsilon = self.epsilon
        n = start
        test_n_steps = len(df) - n if steps==None else steps
        timestamp = df['Timestamp'][n:]
        envRL_test = env.Env(df, full_battery_capacity=self.battery, max_energy=self.max_en, eff=self.eff, price_coefs=self.price_coefs, n_days=self.n_days, n_steps=test_n_steps, low=n, high=len(df), test=True)
        env1 = env.Env(df, full_battery_capacity=self.battery, max_energy=self.max_en, eff=self.eff, price_coefs=self.price_coefs, n_days=self.n_days, n_steps=test_n_steps, low=n, high=len(df), test=True)
        env2 = env.Env(df, full_battery_capacity=self.battery, max_energy=self.max_en, eff=self.eff, price_coefs=self.price_coefs, n_days=self.n_days, n_steps=test_n_steps, low=n, high=len(df), test=True)
        
        
        agent = self.agent
        results = self.run_episode(test_n_steps, envRL_test, env1, env2, agent, epsilon, a, b)
        if self.path == None:
            title = "testing episode"
        else:
            title = "Testing episode: " + self.path
        cost_dqn = results[0]
        cost_comp = results[1]
        print_episode_results(cost_dqn, cost_comp, title, epsilon)
        return results
    
    def run_train_episode(self, n_steps, env0, dqn, epsilon, a, b):
        seed = np.random.randint(0, 1000)
        # ------------------------------------------------------
        env0.reset(seed)
        state = env0.next_observation_normalized()
        cumulative_rewardRL = []
        actions = []
        battery_too_full = []
        rewards = []
        sRL = 0
        # ------------------------------------------------------ 
        for step in range(n_steps):
            action = dqn.choose_action(state, epsilon)
            actions.append(action)
            obs, reward, terminated, data = env0.step(action)
            past_state = state
            state = env0.next_observation_normalized()

            capacity = obs[1, -1]
            past_capacity = obs[1, -2]
            market_price = obs[5, -1]
            cons = obs[3, -1]
            median_market_price = np.median(obs[5,:])
            my_reward = battery_penalty_expand(capacity, env0.full_battery_capacity, 0.1, 0.8) \
                        + a * slope_market_price(capacity, past_capacity, market_price, median_market_price) \
                        + b * reward

            rewards.append(my_reward)
            
            if step == n_steps - 1:
                left_in_battery = obs[1, -1]
                last_price = obs[5, -1]
                left_in_battery_sold = left_in_battery * last_price
                sRL += left_in_battery_sold

            sRL += reward
            cumulative_rewardRL.append(sRL)

            dqn.store_transition(past_state, action, my_reward, state)

            if dqn.memory_counter > self.memory_capacity:
                dqn.learn()

            if terminated:
                break
                
        return sRL
    
    
    def run_episode(self, n_steps, env0, env1, env2, dqn, epsilon, a, b):
        seed = np.random.randint(0, 1000)
        # ------------------------------------------------------
        env0.reset(seed)
        state = env0.next_observation_normalized()
        cumulative_rewardRL = []
        actions = []
        battery_too_full = []
        rewards = []
        sRL = 0
        # ------------------------------------------------------
        env1.reset(seed)
        cumulative_reward_baseline1 = []
        s_baseline1 = 0
        # ------------------------------------------------------
        env2.reset(seed)
        cumulative_reward_baseline2= []
        s_baseline2 = 0
        action_baseline2 = 5
        #-------------------------------------------------------
        en_cost_sum = 0
        en_cost_sums = []
        
        energy_flow_data = []
        
        for step in tqdm(range(n_steps)):
            
            action_baseline1 = step%4
            obs_baseline1, reward_baseline1, terminated_baseline1, _ = env1.step(action_baseline1)
            # -------------------------------------------
            obs_baseline2, reward_baseline2, terminated_baseline2, _ = env2.step(7)
            median_price = np.median(obs_baseline1[5])
            if obs_baseline2[5,-1] > median_price:
                action_baseline2 = 5
            else:
                action_baseline2 = 6
            # -------------------------------------------
            action = dqn.choose_action(state, epsilon)
            actions.append(action)
            obs, reward, terminated, data = env0.step(action)
            energy_flow_data.append(data)
            past_state = state
            state = env0.next_observation_normalized()

            capacity = obs[1, -1]
            past_capacity = obs[1, -2]
            market_price = obs[5, -1]
            cons = obs[3, -1]
            median_market_price = np.median(obs[5,:])
            my_reward = battery_penalty_expand(capacity, env0.full_battery_capacity, 0.1, 0.8) \
                        + a * slope_market_price(capacity, past_capacity, market_price, median_market_price) \
                        + b * reward

            rewards.append(my_reward)
            
            en_cost_sum += cons * market_price * 2
            en_cost_sums.append(en_cost_sum)
            
            # the reinforcement learning tends to save some energy in the battery, therefore we
            # "sell" all the energy left in the battery and add it to the cumulative reward
            if step == n_steps - 1:
                left_in_battery = obs[1, -1]
                last_price = obs[5, -1]
                left_in_battery_sold = left_in_battery * last_price
                sRL += left_in_battery_sold

                left_in_battery_baseline1 = obs_baseline1[1, -1]
                last_price_baseline1 = obs_baseline1[5, -1]
                left_in_battery_sold_baseline1 = left_in_battery_baseline1 * last_price_baseline1
                s_baseline1 += left_in_battery_sold_baseline1

            s_baseline1 += reward_baseline1
            cumulative_reward_baseline1.append(s_baseline1)
            
            s_baseline2 += reward_baseline2
            cumulative_reward_baseline2.append(s_baseline2)

            sRL += reward
            cumulative_rewardRL.append(sRL)

            #dqn.store_transition(past_state, action, my_reward, state)

            #if dqn.memory_counter > self.memory_capacity:
                #dqn.learn()

            if terminated:
                break

        if env0.test:
            df = env0.df
            history = np.array(env0.history)
            steps = history[:, 0]
            battery_capacity = history[:, 1]
            energy_consumption = history[:, 3]
            market_price = history[:, 5] * 10
            amount_paid = history[:, 6]
            time_of_day = history[:, 7]
            colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange'}
            a_colors = [colors[action] for action in actions]
            
            timestamp = list(df['Timestamp'][int(steps[0]):int(steps[-1])+1])
            fig, ax = plt.subplots(2,1,figsize=(15, 15))
            
            ax[0].set_title('System specs: battery size='+ str(self.battery) + 'kWh, max charge energy='+ str(self.max_en))
            ax[1].plot(steps, battery_capacity, label='DQN battery charge', c='blue', linewidth=0.7)
            #ax[1].scatter(steps, battery_capacity, label='actions', c=a_colors, s=4)
            
            cumulative_rewardRL = np.array(cumulative_rewardRL)
            cumulative_rewardRL = - (cumulative_rewardRL - cumulative_rewardRL[0])
            ax[0].plot(steps, cumulative_rewardRL, label='DQN cost', c='darkblue', linewidth=0.7)

            '''
            cmap = plt.get_cmap('Purples')
            colors = ['lightgray', 'white']
            for i in range(n_steps - 1):
                h = ax.get_ylim()[1] - ax.get_ylim()[0]
                w = 1
                c1 = (steps[i], plt.ylim()[0])
                rect = patches.Rectangle(c1, w, h, color=colors[i%2], alpha=0.2)
                ax.add_patch(rect)
            '''
            en_cost_sums = np.array(en_cost_sums)
            ax[0].plot(steps, en_cost_sums, label='no PV, no battery cost', c='red', linewidth=0.7)
            ax[0].grid()
            
            cumulative_reward_baseline1 = np.array(cumulative_reward_baseline1)
            cumulative_reward_baseline1 = - (cumulative_reward_baseline1 - cumulative_reward_baseline1[0])
            history_baseline1 = np.array(env1.history)
            
            ax[0].plot(steps, cumulative_reward_baseline1, label='baseline 1', c='magenta', linewidth=0.7)
            battery_capacity_baseline1 = history_baseline1[:, 1]
            ax[1].plot(steps, battery_capacity_baseline1, label='baseline 1', c='magenta', linewidth=0.7, alpha=0.5)
            
            cumulative_reward_baseline2 = np.array(cumulative_reward_baseline2)
            cumulative_reward_baseline2 = - (cumulative_reward_baseline2 - cumulative_reward_baseline2[0])
            history_baseline2 = np.array(env2.history)
            ax[0].plot(steps, cumulative_reward_baseline2, label='baseline 2', c='violet', linewidth=0.7)
            battery_capacity_baseline2 = history_baseline2[:, 1]
            ax[1].plot(steps, battery_capacity_baseline2, label='baseline 2', c='violet', linewidth=0.7, alpha=0.5)
            
            #ax[2].plot(steps, np.array(rewards), label='reward', linewidth=0.5)
            #ax[2].plot(steps, np.array(amount_paid), label='amount paid', linewidth=0.5)
            
            amount_paid_baseline1 = history_baseline1[:, 6]
            #ax[2].plot(steps, np.array(amount_paid_baseline1), label='amount paid', linewidth=0.5)
            
            amount_paid_baseline2 = history_baseline2[:, 6]
            #ax[2].plot(steps, np.array(amount_paid_baseline2), label='amount paid baseline2', linewidth=0.5)
            
            ax[0].legend(loc='lower left', prop={'size': 8})
            ax[1].legend(loc='lower left', prop={'size': 8})
            #ax[2].legend()
            
            '''
            energy_flow_data = np.array(energy_flow_data).T
            labels = ['gen_old', 'cons_old', 'e_b_out', 'e_b_in', 'e_pv_b', 'e_pv_grid', 'e_b_home', 'e_b_grid', 'e_grid_home', 'e_grid_b']
            
            indices_i_want = [0,1,4,5,9]
            energy_flow_data = energy_flow_data[indices_i_want]
            labels = [labels[i] for i in indices_i_want]
            for line, label in zip(energy_flow_data, labels):
                ax[3].plot(steps, line, label=label)
            
            ax[3].legend()
            '''
            fig.show()
            '''
            costs = np.array([steps, cumulative_rewardRL, cumulative_reward_baseline1, cumulative_reward_baseline2, en_cost_sums, market_price, battery_capacity, battery_capacity_baseline1, battery_capacity_baseline2]).T
            costs = pd.DataFrame(costs, columns=['step', 'rl', 'baseline1', 'baseline2', 'en_cost_sums', 'smp', 'brl', 'bbaseline1', 'bbaseline2'], index=steps)
            costs['timestamp'] = timestamp
            
            a = 7000
            b = 10000
            costs = costs[a:b]
            rl_begin_val = costs['rl'][a]
            costs['rl'] = costs['rl'] - rl_begin_val
            baseline1_begin_val = costs['baseline1'][a]
            costs['baseline1'] = costs['baseline1'] - baseline1_begin_val
            baseline2_begin_val = costs['baseline2'][a]
            costs['baseline2'] = costs['baseline2'] - baseline2_begin_val
            baseline3_begin_val = costs['en_cost_sums'][a]
            costs['en_cost_sums'] = costs['en_cost_sums'] - baseline3_begin_val
            costs.to_csv('costs_with_timestamp.csv')
            '''
            results = [sRL, s_baseline1, s_baseline2, en_cost_sum]
        return results
    
    def load_set_attributes(self, path):
        d = self.load_properties(path)
        self.battery = d["battery"]
        self.max_en = d["power"]
        self.eff = d["efficiency"]
        self.price_coefs = d["price coefficients"]
        self.epsilon = d["epsilon"]
        self.n_days = d["n_days"]
        self.df = pd.read_csv(path + "/df.csv")
        self.agent = dqn.DQN(1,1)
        self.agent.load(path)
        
    
    def save_properties(self, path, df, n_days, epsilon, a, b):
        d = {
            "battery": self.battery,
            "power": self.max_en,
            "efficiency": self.eff,
            "price coefficients": self.price_coefs,
            "n_days": n_days,
            "epsilon": epsilon,
            "a": a,
            "b": b
        }
        df.to_csv(path + '/df.csv')

        with open(path + "/properties.json", "w") as outfile:
                json.dump(d, outfile)

    def load_properties(self, path):
        d = {}       
        with open(path + "/properties.json", "r") as openfile:
            d = json.load(openfile)
        return d
    

    
def battery_penalty_expand(capacity, full_capacity, zero_low, zero_high):
        x = capacity / full_capacity
        if x < zero_low:
            f = - (2 / zero_low * x - 2) ** 2 / 2
        elif x > zero_high:
            f = - (2 / (1 - zero_high) * (x - zero_high)) ** 2 * 4
        else:
            f = 0
        return f + 1
    
def slope_market_price(capacity, past_capacity, market_price, avg_market_price):
    slope = capacity - past_capacity
    relative_market_price = market_price - avg_market_price
    return - (slope * relative_market_price)

def print_episode_results(s1, s2, episode, epsilon):   
    response = "Episode:" + episode + " | Epsilon: " + str(round(epsilon, 2)) + " | DQN cost:" + str(round(s1, 2)) + " | compare cost:" + str(round(s2, 2)) + " | Gain over compare: " + str(round(s1 - s2, 2)) + "( " + str(round(100 * (s1 - s2) / abs(s2), 2)) + "% )"
    print(response)    
    return round(100 * (s1 - s2) / abs(s2), 2), response
    