import numpy as np
# Tesla powerwall has a power of 5.8kW without sunlight
# therefore, a maximum energy of 5.8 * 1/4 = 1,45kWh can be charged or discharged from the battery
# in a 15 minute interval

class Env:
    def __init__(self, df, full_battery_capacity=20, max_energy=6/4, eff=0.9, price_coefs=[2,1], n_days=2, n_steps=1000, low=0, high=30000, test=False):
        self.amount_paid = None
        self.market_price = None
        self.energy_consumption = None
        self.energy_generation = None
        self.ev_consumption = None
        self.current_battery_capacity = None
        self.time_of_day = None
        self.pos = None
        self.df = df
        self.n_steps = n_steps
        self.window_size = 24 * 4 * n_days
        self.low = low
        self.high = high
        self.test = test

        self.full_battery_capacity = full_battery_capacity
        self.max_energy = max_energy
        self.eff = eff
        self.price_coefs = price_coefs
        self.current_step = 0
        self.history = []
        self.reset(0)
        self.state = self.next_observation()

        # query for max and min values in dataframe (used for normalization)
        self.maxs = self.df.max()
        self.mins = self.df.min()

    def reset(self, seed):
        self.current_step = 0
        np.random.seed(seed)
        if self.test:
            self.pos = self.low
            #self.pos = self.pos - (self.pos % self.n_steps)
        else:
            self.pos = np.random.randint(self.low + self.window_size + self.n_steps, self.high - self.n_steps)
            self.pos = self.pos - (self.pos % self.n_steps)  # = 1 week
        self.current_battery_capacity = np.array([0] * self.window_size)
        self.energy_generation = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'Energy_Generation'])
        self.energy_consumption = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'Energy_Consumption'])
        self.ev_consumption = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'EV_Consumption'])
        self.market_price = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1, 'SMP'])
        self.amount_paid = np.array([0] * self.window_size)
        self.time_of_day = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'Time_of_Day'])
        self.history = []
        return self.next_observation()

    def next_observation(self):
        return np.array([
            [self.current_step] * self.window_size,
            self.current_battery_capacity,
            self.energy_generation,
            self.energy_consumption,
            self.ev_consumption,
            self.market_price,
            self.amount_paid,
            self.time_of_day
        ])

    def next_observation_normalized(self):
        current_battery_capacity = self.current_battery_capacity / self.full_battery_capacity
        energy_generation = (self.energy_generation - self.mins['Energy_Generation']) / (
                self.maxs['Energy_Generation'] - self.mins['Energy_Generation'])
        energy_consumption = (self.energy_consumption - self.mins['Energy_Consumption']) / (
                self.maxs['Energy_Consumption'] - self.mins['Energy_Consumption'])
        ev_consumption = (self.ev_consumption - self.mins['EV_Consumption']) / (
                self.maxs['EV_Consumption'] - self.mins['EV_Consumption'])
        
        if self.maxs['SMP'] - self.mins['SMP'] == 0:
            market_price = self.market_price
        else:
            market_price = ((self.market_price - self.mins['SMP']) / (self.maxs['SMP'] - self.mins['SMP']))
        # amount paid = [market price] * [MWh bought from the grid]. Therefore max possible value equals
        # [max market price] * [max consumption]
        # and min possible value is 0 (if we take all the energy from the battery
        amount_paid = (self.amount_paid / (self.maxs['SMP'] * self.maxs['Energy_Consumption']))
        time_of_day = self.time_of_day
        return np.array([
            current_battery_capacity,
            energy_generation,
            energy_consumption,
            ev_consumption,
            market_price,
            amount_paid,
            time_of_day
        ]).flatten()

    def step(self, action):
        data = self.take_action(action)

        self.current_step += 1

        self.energy_generation = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'Energy_Generation'])
        self.energy_consumption = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'Energy_Consumption'])
        self.ev_consumption = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'EV_Consumption'])
        self.market_price = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1, 
            'SMP'])
        self.time_of_day = np.array(
            self.df.loc[self.current_step + self.pos - self.window_size:self.current_step + self.pos - 1,
            'Time_of_Day'])

        reward = - self.amount_paid[self.window_size - 1]
        terminated = self.current_step >= self.n_steps
        obs = self.next_observation()
        self.history.append(obs[:, self.window_size - 1])
        return obs, reward, terminated, data

    def take_action(self, action):
        cons = self.energy_consumption[self.window_size - 1]
        ev_consumption = self.ev_consumption[self.window_size - 1]
        gen = self.energy_generation[self.window_size - 1]
        smp = self.market_price[self.window_size - 1]
        b = self.current_battery_capacity[self.window_size - 1]
        b_max = self.max_energy
        b_new = None
        amount_paid_now = None
        p_sell = smp * self.price_coefs[1]
        p_buy = smp * self.price_coefs[0]
        
        e_pv_home = min(gen, cons)
        cons_old = cons
        gen_old = gen
        cons = cons - e_pv_home
        gen = gen - e_pv_home
        
        e_b_out = min(b_max, b) * self.eff
        e_b_in = min(b_max, self.full_battery_capacity - b) / self.eff
        
        e_pv_b = 0
        e_pv_grid = 0
        
        e_b_home = 0
        e_b_grid = 0
        
        e_grid_home = 0
        e_grid_b = 0
        
        if action == 0:
            e_pv_b = min(e_b_in, gen)
            e_grid_b = e_b_in - e_pv_b
            e_grid_home = cons
            e_pv_grid = gen - e_pv_b
        
            b_new = b + (e_pv_b + e_grid_b) * self.eff
            amount_paid_now = (e_grid_home + e_grid_b) * p_buy - e_pv_grid * p_sell
        
        elif action == 1:
            e_b_home = min(e_b_out, cons)
            e_grid_home = cons - e_b_home
            e_pv_grid = gen
            e_b_grid = e_b_out - e_b_home
            
            b_new = b - e_b_home / self.eff
            amount_paid_now = e_grid_home * p_buy - (e_pv_grid + e_b_grid) * p_sell
            
        elif action == 2:
            e_grid_home = cons
            e_pv_b = min(e_b_in, gen)
            e_pv_grid = gen - e_pv_b
            
            b_new = b + e_pv_b * self.eff
            amount_paid_now = e_grid_home * p_buy - e_pv_grid * p_sell
                
        elif action == 3:
            e_b_home = min(e_b_out, cons)
            e_grid_home = cons - e_b_home
            e_pv_grid = gen
            
            b_new = b - e_b_home / self.eff
            amount_paid_now = e_grid_home * p_buy - e_pv_grid * p_sell
        

        elif action == 4:
            # things go out of the battery
            amount_paid_now, b_new = self.action0(b, b_max, gen_old, cons_old, p_buy, p_sell)
        
        elif action == 5:
            # things go into the battery            
            amount_paid_now, b_new = self.action1(b, b_max, gen_old, cons_old, p_buy, p_sell)

        elif action == 6:
            # no change in battery
            if gen > cons:
                e_pv_home = cons
                e_pv_grid = gen - cons
                e_grid_home = 0
            else:
                e_pv_home = gen
                e_pv_grid = 0
                e_grid_home = cons - gen
                
            b_new = b
            cost = - e_pv_grid * p_sell + e_grid_home * p_buy
            amount_paid_now = cost
        
        elif action == 7:
            if gen > cons:
                e_pv_grid = gen - cons
                cost = - e_pv_grid * p_sell
            else:
                e_grid_home = cons - gen
                cost = e_grid_home * p_buy
            
            amount_paid_now = cost
            b_new = 0
        
        self.amount_paid = np.concatenate([self.amount_paid[1:self.window_size], [amount_paid_now]])
        self.current_battery_capacity = np.concatenate(
            [self.current_battery_capacity[1:self.window_size], [b_new]])
        
        return [gen_old, cons_old, e_b_out, e_b_in, e_pv_b, e_pv_grid, e_b_home, e_b_grid, e_grid_home, e_grid_b]
    
    def action0(self, b, b_max, gen, cons, p_buy, p_sell):
        # things go out of the battery
        e_b_out = min(b_max, b) * self.eff
            
        if gen > cons:
                
            e_pv_home = cons
            e_pv_grid = gen - cons
                
            e_b_home = 0
            e_b_grid = e_b_out
                
            e_grid_home = 0
                
        else:
                
            e_pv_home = gen
            e_pv_grid = 0
                
            if e_b_out > cons - gen:
                    
                e_b_home = cons - gen
                e_b_grid = e_b_out - (cons - gen)
                    
                e_grid_home = 0
                
            else:
                    
                e_b_home = e_b_out
                e_b_grid = 0
                    
                e_grid_home = cons - gen - e_b_out
            
        new_b = b - e_b_out / self.eff
        cost = - (e_pv_grid + e_b_grid) * p_sell + e_grid_home * p_buy
        
        return cost, new_b
        
        
    def action1(self, b, b_max, gen, cons, p_buy, p_sell):
        # things go into the battery
        e_b_in = min(b_max, self.full_battery_capacity - b) / self.eff
            
        e_b_grid = 0
            
        if gen > e_b_in:
                
            e_pv_b = e_b_in
            e_grid_b = 0
                
            if gen > e_b_in + cons:
                    
                e_pv_home = cons
                e_pv_grid = gen - e_b_in - cons
                e_grid_home = 0
                    
            else:
                    
                e_pv_home = gen - e_b_in
                e_pv_grid = 0
                e_grid_home = cons - (gen - e_b_in)
                
        else:
                
            e_pv_b = gen
            e_grid_b = e_b_in - gen
            e_pv_home = 0
            e_pv_grid = 0
            e_grid_home = cons
                
                
        new_b = b + e_b_in * self.eff
        cost = - (e_pv_grid + e_b_grid) * p_sell + (e_grid_home + e_grid_b) * p_buy
            
        return cost, new_b
        
            
    def render(self):
        return None
        # print('Step:', self.current_step, '| Current battery capacity:', self.current_battery_capacity, '| Amount paid:', self.amount_paid)

        
def negative(t):
    for item, i in enumerate(t):
        if item < 0:
            print(i)