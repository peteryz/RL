import gym
import random
import os
import numpy as np
from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam
import logging
from math import sqrt
import csv
class Agent():
    def __init__(self, state_size, action_size):
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=20000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.00
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state):
        print "-----------------------------------------------"
        #print "***********"
        if np.random.rand() <= self.exploration_rate:
            #print "RANDOM ACT"
            #print "***********"
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        #print "ACT VALUES: ", act_values
        #print "***********"
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


def writeData(time_step, sup_reward, ret_reward, sup_freshness, ret_freshness, fulfilled, sup_waste, ret_waste, sup_loss, ret_loss, sup_action, ret_action, sup_inv, ret_inv, demand, orders, flow):
    # time step, s_reward, r_reward, s_loss, r_loss, s_demand, r_demand, s_inv, r_inv, s_flow, r_flow
    # s_freshness, r_freshness, s_fulfilled, r_fulfilled, s_waste, r_waste, s_action, r_action, s_order, r_order
    csvfile = open('data_dual_v2.csv', 'a')
    sup_order = orders[0,1]
    ret_order = orders[1,2]
    flow_to_sup = sum(flow[0,1])
    flow_to_ret = sum(flow[1,2])
    sup_demand = orders[1,2]
    ret_demand = demand[2]
    sup_flow = sum(flow[0,1])
    ret_flow = sum(flow[1,2])
    sup_fulfilled = fulfilled[1]
    ret_fulfilled = fulfilled[2]
    with csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerow([repr(time_step)] + [repr(sup_reward)] + [repr(ret_reward)] + [repr(sup_loss)]
         + [repr(ret_loss)] + [repr(sup_demand)] + [repr(ret_demand)] + [repr(sup_inv)] + [repr(ret_inv)]
         + [repr(sup_flow)] + [repr(ret_flow)] + [repr(sup_freshness)] + [repr(ret_freshness)]
         + [repr(sup_fulfilled)] + [repr(ret_fulfilled)] + [repr(sup_waste)] + [repr(ret_waste)]
         + [repr(sup_action)] + [repr(ret_action)] + [repr(sup_order)] + [repr(ret_order)])

class Supplier:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 10000

        self.state_size        = 21
        self.action_size       = 4
        self.agent             = Agent(self.state_size, self.action_size)

    def getAction(self, state):
        action = self.agent.act(state)
        return action

    def updateState(self, state, action, reward, inv, pipe):
        done = False
        next_state = np.concatenate((inv[1], pipe[0,1], pipe[1,1]), axis=0)
        next_state = np.reshape(next_state, [1, self.state_size])
        self.agent.remember(state, action, reward, next_state, done)
        return next_state

    def updateModel(self):
        self.agent.replay(self.sample_batch_size)


    def get_base_stock_level(self, A, dparams, lead, i):
        z = 2
        for parent in range(0, A.shape[0]):
            if A[parent, i] == 1:
                return dparams[0] * (lead[parent, i] + 1) + (z * dparams[1]) * (sqrt(lead[parent, i] + 1))

class Retailer:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 10000

        self.state_size        = 21
        self.action_size       = 4
        self.agent             = Agent(self.state_size, self.action_size)


    def getAction(self, state):
        action = self.agent.act(state)
        return action

    def updateState(self, state, action, reward, inv, pipe):
        done = False
        next_state = np.concatenate((inv[2], pipe[0,2], pipe[1,2]), axis=0)
        next_state = np.reshape(next_state, [1, self.state_size])
        self.agent.remember(state, action, reward, next_state, done)
        return next_state

    def updateModel(self):
        self.agent.replay(self.sample_batch_size)

    def get_base_stock_level(self, A, dparams, lead, i):
        z = 2
        for parent in range(0, A.shape[0]):
            if A[parent, i] == 1:
                return dparams[0] * (lead[parent, i] + 1) + (z * dparams[1]) * (sqrt(lead[parent, i] + 1))

                
def pull(orders, n, m, A, lead, inv, pipe, dparams, yparams, source, sink):  
    waste_sup = 0
    waste_ret = 0
    loss_sup = 0
    loss_ret = 0
    d = np.zeros(n)
    fulfilled = np.zeros(n)
    unfulfilled = d
    
    # Initialize a random d
    d = np.maximum(0, np.random.normal(dparams[0], dparams[1], n))
    d = np.minimum(d, sink * (dparams[0]+dparams[1]*10))
    #d[1] = orders[1,2]
    print "DEMAND FROM CUSTOMERS: ", d[2]
    print "DEMAND FROM RETAILER: ", orders[1,2]
    logging.debug("Demand vector:\n %s \n", str(d))
    
    # Initialize yield vector
    prod = np.zeros((n,m))
    prod[:,m-1] = np.random.normal(yparams[0], yparams[1], n)
    prod[:,m-1] = np.minimum(prod[:,m-1], source * (yparams[0]+yparams[1]*10))
    
    logging.debug("Production vector:\n %s \n", str(prod))
    
    # Set base stock level based on service level, lead time, every node
    logging.debug("Lead time matrix:\n %s \n", str(lead))
    l = np.amin(lead,0) # min lead time for each node
    
    ## Calculate the minimum freshness a node i needs to send to downstream
    # Roughly speaking, we use the notion of shortest path to sink nodes
    toSink = np.ones(n,dtype=np.int16) * 1000
    front = np.copy(sink) 
    for i in range(n):
        if sink[i] == 1:
            toSink[i] = 0
    while front.any() > 0:
        copyFront = np.copy(front)
        for j in np.nonzero(copyFront)[0]:
            front[j] = 0
            for i in np.nonzero(A[:,j])[0]:
                front[i] = 1
                toSink[i] = min(toSink[i], toSink[j] + lead[i,j])
    
    # Converting orders[i,j] to flows[i,j,k]
    # Generate flow as much as possible, up to the upstream on-hand inv level
    # Assumptions: 
    #   FIFO, 
    #   Proportional fulfillment of orders
    #   Inv used only when there's enough life left to survive the lead time
    flow = np.zeros((n,n,m)) # flow[i,j,k]: amount of life-k flow from i to j
    for i in range(n):
        for j in range(n):
            if A[i,j] == 1:
                if source[i] == 1:
                    flow[i,j,m-1] = orders[i,j]
                else:
                    #l = lead[i,j] + 1
                    l = lead[i,j] + toSink[j] + 1
                    lastK = l
                    for k in range(m):
                        if k >= l:
                            if sum(inv[i,l:k+1])<=sum(orders[i,:]) \
                            and sum(orders[i,:]) > 0:
                                flow[i,j,l:k+1] = 1.0 * inv[i,l:k+1] * \
                                orders[i,j] / sum(orders[i,:])
                                lastK += 1
                    if lastK < m and lastK >= l and sum(orders[i,:]) > 0:
                        flow[i,j,lastK] = 1.0 * (sum(orders[i,:]) \
                        - sum(inv[i,l:lastK])) * orders[i,j] / sum(orders[i,:])
    print "FLOW FARM -> SUP, SUP -> RET: ", sum(flow[0,1]), ", ", sum(flow[1,2])
    
    # Transform d to fill array -- the actual demand fulfilled
    # fill[i,k] amount of life-k inventory being used by i to fulfill demand
    fill = np.zeros((n,m))
    for i in range(n):
        lastK = 0
        for k in range(m):
            if sum(inv[i,:k+1]) <= d[i]:
                fill[i,:k+1] = inv[i,:k+1]
                lastK += 1
        if lastK < m and d[i] > 0:
            fill[i,lastK] = d[i] - sum(inv[i,:lastK])
    logging.debug("Demand fulfilled:\n %s \n", str(fill))
    print "ORDERS FULFILLED: SUP, RET", sum2(flow[1,:,:]), ", ", sum(fill[2])

    # Disposal decision
    # No intentionally disposal for this basic pull
    disp = np.zeros((n,m))
    
    # Call update function and accumulate waste, etc.
    logging.debug("Calling update function. \n")
    inv, pipe, waste_s, waste_r = update(n, m, A, lead, inv, pipe, 
                                    prod, flow, fill, disp, source, sink)
    logging.debug("Finished update function. \n")
    
    waste_sup += waste_s
    waste_ret += waste_r
    
    fulfilled = np.sum(flow, axis=(1,2)) + sum(fill.T).T # fulfilled [i] is the amount shipped out of i to fulfill demand/downstream order
    unfulfilled = sum(orders.T).T + d - fulfilled # unfulfilled[i] is the demand loss or unfilled order from i to demand/downstream
    loss = sum(unfulfilled)
    loss_sup = unfulfilled[1]
    loss_ret = unfulfilled[2]
    
    #print "fulfilled sum=", sum(fulfilled)
    freshness_sup = 0.0
    freshness_ret = 0.0
    if sum(fulfilled)  < 0.0001:
        freshness = 0
    else:
        freshness = np.dot(np.sum(fill,axis=0), np.arange(m)+1) / m / sum(fulfilled)

    if fulfilled[1] < 0.0001:
        freshness_sup = 0
    else:
        freshness_sup = np.dot(np.sum(flow[1,:,:],axis=0), np.arange(m)+1) / m / fulfilled[1]
    
    if fulfilled[2] < 0.0001:
        freshness_ret = 0
    else:
        freshness_ret = np.dot(fill[2,:], np.arange(m)+1) / m / fulfilled[2]

    logging.debug("Total demand filled: %d", sum2(fill))
    logging.debug("Total demand unfilled: %d", loss)
    
    logging.debug("Pipe \n %s ", str(pipe))
    
    return inv, pipe, waste_sup, waste_ret, loss_sup, loss_ret, freshness_sup, freshness_ret, fulfilled, unfulfilled, d, flow

def update(n, m, A, lead, inv, pipe, prod, flow, fill, disp, source, sink):
    #waste = 0
    waste_ret = 0
    waste_sup = 0

    negTolerance = -0.001

    logging.debug("Inventory on hand:\n %s", str(inv))
    logging.debug("Pipeline inventory pipe[l,i,k]:\n %s \n", str(pipe))
    
    # 1. closest pipeline inventory arriving
    inv += pipe[0]

    #print "new inventory w/ pipe: ", inv

    logging.debug("Inventory update: received from pipeline\n %s \n", str(inv))
    if np.min(inv) < negTolerance:
        logging.warning("Inv below 0 after receiving pipe: \n %s", str(inv))
    
    # 2. pipeline inventory aging and moving closer
    pipe = np.roll(pipe, -1, axis=0) # aging
    pipe[-1:,:,:] = 0
    
    pipe = np.roll(pipe, -1, axis=2) # expiring
    pipe[:,:,-1:] = 0
    
    logging.debug("Pipeline update: aging\n %s \n", str(pipe))
    if np.min(pipe) < negTolerance:
        logging.warning("Pipe below 0 after pipe aging: \n %s", str(pipe))
    
    # Orders are transformed into flow vectors
    # 3a. inflow from upstream to pipeline inventory
    rows, cols = np.nonzero(A)
    for i in range(len(rows)):
        l = lead[rows[i],cols[i]]
        pipe[l-1][cols[i]] += np.sum(flow[:,cols[i],:],axis=0)
    logging.debug("Pipeline update: receiving from upstream\n %s \n",str(pipe))
    if np.min(pipe) < negTolerance:
        logging.warning("Pipe below 0 after receiving flow: \n %s", str(pipe))

    # 3b. outflow: shipment to downstream
    inv -= np.sum(flow,axis=1)
    logging.debug("Inventory update: sending to downstream\n %s \n", str(inv))
    if np.min(inv) < negTolerance and np.argmin(inv) >= m:
        logging.warning("Inv below 0 after outflow: \n %s", str(inv))
    
    # 4. production arrival 
    inv += prod
    logging.debug("Productions:\n %s \n", str(prod))
    logging.debug("Inventory udpate: production arrival \n %s \n", str(inv))
    
    # 5. outflow: demand fulfillment (at sink nodes)
    inv -= fill
    logging.debug("Demand quantity to serve:\n %s \n", str(fill))
    logging.debug("Inventory update: demand fulfillment\n %s \n", str(inv))
    if np.min(inv) < negTolerance:
        logging.warning("Inv below 0 after demand fulfill: \n %s", str(inv))
    
    # 6. disposal
    inv -= disp
    waste_sup += sum(disp[1])
    waste_ret += sum(disp[2])
    #waste += sum2(disp)
    logging.debug("Disposal:\n %s \n", str(disp))
    logging.debug("Inventory update: disposal\n %s \n", str(inv))
    if np.min(inv) < negTolerance:
        logging.warning("Inv below 0 after disposal: \n %s", str(inv))
    
    # 7. on-hand inventory aging
    
    # Assumptions: the source nodes are just holders for inv, wastes there 
    # are not included. We only include wastes on things that non-source nodes
    # ordered.
    #waste += sum2(inv[:,-1:])
    
    
    waste_sup += inv[1,0]
    waste_ret += inv[2,0]

    inv = np.roll(inv, -1, axis=1)
    inv[:,-1:] = 0
    logging.debug("Inventory update: aging\n %s \n", str(inv))

    return inv, pipe, waste_sup, waste_ret


def sum2(array):
    return sum(sum(array))

def sum3(array):
    return sum(sum(sum(array)))

def generateOrders(base_stock_supplier, base_stock_retailor, action_ret, action_sup, inv, pipe):
    current_inv_supplier = sum(inv[1]) + sum(pipe[0,1]) + sum(pipe[1,1])
    current_inv_retailor = sum(inv[2]) + sum(pipe[0,2]) + sum(pipe[1,2])
    order_amount_supplier = 0
    order_amount_retailor = 0
        
    act0 = 1.0/5.0
    act1 = 2.0/5.0
    act2 = 3.0/5.0
    act3 = 4.0/5.0


    ret_ord0 = (base_stock_retailor * act0) - current_inv_retailor
    ret_ord1 = (base_stock_retailor * act1) - current_inv_retailor
    ret_ord2 = (base_stock_retailor * act2) - current_inv_retailor
    ret_ord3 = (base_stock_retailor * act3) - current_inv_retailor

    sup_ord0 = (base_stock_supplier * act0) - current_inv_supplier
    sup_ord1 = (base_stock_supplier * act1) - current_inv_supplier
    sup_ord2 = (base_stock_supplier * act2) - current_inv_supplier
    sup_ord3 = (base_stock_supplier * act3) - current_inv_supplier

    orders = np.zeros((3,3))

    if action_sup == 0:
        order_amount_supplier = sup_ord0
    elif action_sup == 1:
        order_amount_supplier = sup_ord1
    elif action_sup == 2:
        order_amount_supplier = sup_ord2
    elif action_sup == 3:
        order_amount_supplier = sup_ord3

    if action_ret == 0:
        order_amount_retailor = ret_ord0
    elif action_ret == 1:
        order_amount_retailor = ret_ord1
    elif action_ret == 2:
        order_amount_retailor = ret_ord2
    elif action_ret == 3:
        order_amount_retailor = ret_ord3
    
    # make sure we don't order negative amounts
    if order_amount_retailor < 0:
        order_amount_retailor = 0
    if order_amount_supplier < 0:
        order_amount_supplier = 0
        
    # add orders to array
    orders[0,1] = order_amount_supplier
    orders[1,2] = order_amount_retailor

    #print "supplier order amount = ", order_amount
    return orders

def run(n, m, A, lead, inv, pipe, dparams, yparams, source, sink):
    # file to write data into
    csvfile = open('data_dual_v3.csv', 'a')
    with csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerow(["Time Step"] + ["Sup Reward"] + ["Ret Reward"] + ["Sup Loss"] + ["Ret Loss"] + ["Sup Demand"] + ["Ret Demand"] + ["Sup Inv"] + ["Ret Inv"] + ["Sup Flow"] + ["Ret Flow"] + ["Sup freshness"] + ["Ret freshness"] + ["Sup fulfilled"] + ["Ret fulfilled"] + ["Sup Waste"] + ["Ret Waste"] + ["Sup Action"] + ["Ret Action"] + ["Sup Order"] + ["Ret Order"])
        # time step, s_reward, r_reward, s_loss, r_loss, s_demand, r_demand, s_inv, r_inv, s_flow, r_flow
        # s_freshness, r_freshness, s_fulfilled, r_fulfilled, s_waste, r_waste, s_action, r_action, s_order, ret_order

    
    #initalize states
    sup_state = np.array([0] * 21) 
    sup_state = np.reshape(sup_state, [1, 21])
    ret_state = np.array([0] * 21) 
    ret_state = np.reshape(ret_state, [1, 21])
    
    #initalize suppler and retailer
    supplier = Supplier()
    retailer = Retailer()

    #set base stock levels
    sup_base_stock = supplier.get_base_stock_level(A, dparams, lead, 1)
    ret_base_stock = retailer.get_base_stock_level(A, dparams, lead, 2)

    time_step = 0
    while True:
        for j in range(70):
            # get orders
            sup_action = supplier.getAction(sup_state)
            ret_action = retailer.getAction(ret_state)
            orders = generateOrders(sup_base_stock, ret_base_stock, ret_action, sup_action, inv, pipe)
            
            # print debugging
            print "ACTIONS: ", sup_action, ", ", ret_action
            print "ORDERS: ", orders[0,1], ", ", orders[1,2]
            print "INV: ", sum(inv[1]), ", ", sum(inv[2])
            pipe_sum_sup = (sum(pipe[0,1]) + sum(pipe[1,1]))
            pipe_sum_ret = (sum(pipe[0,2]) + sum(pipe[1,2]))
            print "PIPE: ", pipe_sum_sup, ", ", pipe_sum_ret

            # get next state and reward
            inv, pipe, sup_waste, ret_waste, sup_loss, ret_loss, sup_freshness, ret_freshness, fulfilled, unfulfilled, d, flow = pull(orders, n, m, A, lead, inv, pipe, dparams, yparams, source, sink)
            sup_reward = (fulfilled[1] * sup_freshness) - ((0.3 * sup_waste)+ sup_loss)
            ret_reward = (fulfilled[2] * ret_freshness) - ((0.3 * ret_waste)+ ret_loss)
            sup_state = supplier.updateState(sup_state, sup_action, sup_reward, inv, pipe)
            ret_state = retailer.updateState(ret_state, ret_action, ret_reward, inv, pipe)

            # print debugging
            print "LOSS: ", sup_loss, ", ", ret_loss
            print "FRESHNESS: ", sup_freshness, ", ", ret_freshness
            print "REWARD: ", sup_reward, ", ", ret_reward
            writeData(time_step, sup_reward, ret_reward, sup_freshness, ret_freshness, fulfilled, sup_waste, ret_waste, sup_loss, ret_loss, sup_action, ret_action, sum(inv[1]), sum(inv[2]), d, orders, flow)
            time_step += 1
        # update models
        supplier.updateModel()
        retailer.updateModel()


def main():
    n = 3 #3 nodes in the system
    m = 7 #shelf life
    A = np.zeros((n,n))
    A[0,1] = 1 #farm feeds supplier
    A[1,2] = 1 #supplier feeds farm
    lead = np.zeros((n,n),dtype=np.int16) + 1000
    lead[0,1] = 2 #lead time from farm to supplier
    lead[1,2] = 2 #lead time from supplier to retailor
    maxLead = 2
    source = np.zeros(n)
    sink = np.zeros(n)
    source[0] = 1 #farm
    sink[2] = 1 #retailor
    dparams = np.array([10,5]) #demand distribution
    yparams = np.array([100,2]) #yield distribution
    inv = np.maximum(np.random.rand(n,m), 0)
    pipe = np.maximum(np.random.rand(maxLead, n, m), 0)
    
    #initialize farm inventory
    for i in range(7):
        inv[0,i] = 1000

    #initialize farm pipe
    for j in range(7):
        pipe[0,0,j] = 1000 #1 day away
    for k in range(7):
        pipe[1,0,k] = 1000 #2 days away


    run(n, m, A, lead, inv, pipe, dparams, yparams, source, sink)

if __name__ == "__main__":
    main()


    '''
    convert state space to 21 + 21 = 42
    order[i,j] = base_stock.- (inv + sum(pipe[:,j])) where. i = source, j = dest
    '''

