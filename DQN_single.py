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
        print "***********"
        if np.random.rand() <= self.exploration_rate:
            print "RANDOM ACT"
            print "***********"
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        #print "ACT VALUES: ", act_values
        print "***********"
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

def whichAction(action):
    # action = supplier_action, retailor_action
    # 0 = 0,0
    # 1 = 0,1
    # 2 = 0,2
    # 3 = 0,3
    # 4 = 1,0
    # 5 = 2,0
    # 6 = 3,0
    # 7 = 1,1
    # 8 = 2,2
    # 9 = 3,3
    # 10 = 1,2
    # 11 = 2,1
    # 12 = 1,3
    # 13 = 3,1
    # 14 = 2,3
    # 15 = 3,2  

    if action == 0:
        return "0,0"
    elif action == 1:
        return "0,1"
    elif action == 2:
        return "0,2"
    elif action == 3:
        return "0,3"
    elif action == 4:
        return "1,0"
    elif action == 5:
        return "2,0"
    elif action == 6:
        return "3,0"
    elif action == 7:
        return "1,1"
    elif action == 8:
        return "2,2"
    elif action == 9:
        return "3,3"
    elif action == 10:
        return "1,2"
    elif action == 11:
        return "2,1"
    elif action == 12:
        return "1,3"
    elif action == 13:
        return "3,1"
    elif action == 14:
        return "2,3"
    elif action == 15:
        return "3,2"

def writeData(i, reward, freshness, fulfilled, waste, loss, action, sup_inv, ret_inv, demand, orders, flow):
    csvfile = open('data_normal_v1.2.csv', 'a')
    sup_order = orders[0,1]
    ret_order = orders[1,2]
    flow_to_sup = sum(flow[0,1])
    flow_to_ret = sum(flow[1,2])
    with csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerow([repr(reward)] + [repr(freshness)] + [repr(fulfilled)] + [repr(waste)] + [repr(loss)] + [repr(action)] + [repr(sup_inv)] + [repr(ret_inv)] + [repr(demand)] + [repr(sup_order)] + [repr(ret_order)] + [repr(flow_to_sup)] + [repr(flow_to_ret)] + [repr(i)])



class Supplier:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 10000

        self.state_size        = 42
        self.action_size       = 16
        self.agent             = Agent(self.state_size, self.action_size)

    def run(self, n, M, A, lead, inv, pipe, dparams, yparams, source, sink):
        while True:
            state = np.array([0] * 42) 
            state = np.reshape(state, [1, self.state_size])
            i = 0
            done = False
            call_replay = 0
            while not done:
                if call_replay < 70:
                    action = self.agent.act(state)
                    base_stock_supplier = int(self.get_base_stock_level(A, dparams, lead, 1))
                    base_stock_retailor = int(self.get_base_stock_level(A, dparams, lead, 2))
                    orders = self.generateOrders(base_stock_supplier, base_stock_retailor, action, inv, pipe)
                    
                    
                    pipe_sum_sup = (sum(pipe[0,1]) + sum(pipe[1,1]))
                    pipe_sum_ret = (sum(pipe[0,2]) + sum(pipe[1,2]))
                    print "ACTION: ", whichAction(action)
                    print "SUPPLIER INV, PIPE, ORDER: ", sum(inv[1]), ", ", pipe_sum_sup, ", ", orders[0,1]
                    print "RETAILER INV, PIPE, ORDER: ", sum(inv[2]), ", ", pipe_sum_ret, ", ", orders[1,2]



                    inv, pipe, waste, loss, freshness, fulfilled, unfulfilled, d, flow = pull(orders, n, M, A, lead, inv, pipe, dparams, yparams, source, sink)
                    next_state = np.concatenate((inv[1], pipe[0,1], pipe[1,1], inv[2], pipe[0,2], pipe[1,2]), axis=0)
                    reward = (fulfilled[2] * freshness) - ((0.3 * waste)+ loss)
                    
                    print "FRESHNESS: ", freshness
                    print "REWARD: ", reward

                    sup_inv_sum = sum(inv[1]) + sum(pipe[0,1]) + sum(pipe[1,1])
                    ret_inv_sum = sum(inv[2]) + sum(pipe[0,2]) + sum(pipe[1,2])
                    writeData(i, reward, freshness, fulfilled[2], waste, loss, action, sup_inv_sum, ret_inv_sum, d[2], orders, flow)
                    i += 1

                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    call_replay += 1
                self.agent.replay(self.sample_batch_size)
                call_replay = 0

    def get_base_stock_level(self, A, dparams, lead, i):
        """
        :param i: current node with parent k
        :param A: Adjacency matrix for node
        :return: base stock level for node i
        """
        z = 2
        for parent in range(0, A.shape[0]):
            if A[parent, i] == 1:
                return dparams[0] * (lead[parent, i] + 1) + (z * dparams[1]) * (sqrt(lead[parent, i] + 1))

    def generateOrders(self, base_stock_supplier, base_stock_retailor, action, inv, pipe):
        current_inv_supplier = 0
        current_inv_retailor = 0
        order_amount_supplier = 0
        order_amount_retailor = 0
        current_inv_supplier = sum(inv[1]) + sum(pipe[0,1]) + sum(pipe[1,1])
        current_inv_retailor = sum(inv[2]) + sum(pipe[0,2]) + sum(pipe[1,2])
        
        # action = supplier_action, retailor_action
        # 0 = 0,0
        # 1 = 0,1
        # 2 = 0,2
        # 3 = 0,3
        # 4 = 1,0
        # 5 = 2,0
        # 6 = 3,0
        # 7 = 1,1
        # 8 = 2,2
        # 9 = 3,3
        # 10 = 1,2
        # 11 = 2,1
        # 12 = 1,3
        # 13 = 3,1
        # 14 = 2,3
        # 15 = 3,2

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

        #print ret_ord0, ret_ord1, ret_ord2, ret_ord3
        #print sup_ord0, sup_ord1, sup_ord2, sup_ord3

        orders = np.zeros((3,3))
        if action == 0:
            order_amount_supplier = sup_ord0
            order_amount_retailor = ret_ord0
        elif action == 1:
            order_amount_supplier = sup_ord0
            order_amount_retailor = ret_ord1
        elif action == 2:
            order_amount_supplier = sup_ord0
            order_amount_retailor = ret_ord2
        elif action == 3:
            order_amount_supplier = sup_ord0
            order_amount_retailor = ret_ord3
        elif action == 4:
            order_amount_supplier = sup_ord1
            order_amount_retailor = ret_ord0
        elif action == 5:
            order_amount_supplier = sup_ord2
            order_amount_retailor = ret_ord0
        elif action == 6:
            order_amount_supplier = sup_ord3
            order_amount_retailor = ret_ord0
        elif action == 7:
            order_amount_supplier = sup_ord1
            order_amount_retailor = ret_ord1
        elif action == 8:
            order_amount_supplier = sup_ord2
            order_amount_retailor = ret_ord2
        elif action == 9:
            order_amount_supplier = sup_ord3
            order_amount_retailor = ret_ord3
        elif action == 10:
            order_amount_supplier = sup_ord1
            order_amount_retailor = ret_ord2
        elif action == 11:
            order_amount_supplier = sup_ord2
            order_amount_retailor = ret_ord1
        elif action == 12:
            order_amount_supplier = sup_ord1
            order_amount_retailor = ret_ord3
        elif action == 13:
            order_amount_supplier = sup_ord3
            order_amount_retailor = ret_ord1
        elif action == 14:
            order_amount_supplier = sup_ord2
            order_amount_retailor = ret_ord3
        elif action == 15:
            order_amount_supplier = sup_ord3
            order_amount_retailor = ret_ord2
        
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
                
def pull(orders, n, m, A, lead, inv, pipe, dparams, yparams, source, sink):  
    waste = 0
    loss = 0
    d = np.zeros(n)
    fulfilled = np.zeros(n)
    unfulfilled = d
    
    # Initialize a random d
    d = np.maximum(0, np.random.normal(dparams[0], dparams[1], n))
    d = np.minimum(d, sink * (dparams[0]+dparams[1]*10))
    
    print "DEMAND FROM STORE: ", sum(d)
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
    print "DEMAND FULFILLED: SUP, RET", sum(fill[1]), ", ", sum(fill[2])

    # Disposal decision
    # No intentionally disposal for this basic pull
    disp = np.zeros((n,m))
    
    # Call update function and accumulate waste, etc.
    logging.debug("Calling update function. \n")
    inv, pipe, newWaste = update(n, m, A, lead, inv, pipe, 
                                    prod, flow, fill, disp, source, sink)
    logging.debug("Finished update function. \n")
    
    waste += newWaste
    
    fulfilled = sum(fill.T).T
    unfulfilled = d - fulfilled
    loss = sum(unfulfilled)
    
    #print "fulfilled sum=", sum(fulfilled)
    if sum(fulfilled)  < 0.0001:
        freshness = 0
    else:
        freshness = np.dot(np.sum(fill,axis=0), np.arange(m)+1) / m / sum(fulfilled)
    
    logging.debug("Total demand filled: %d", sum2(fill))
    logging.debug("Total demand unfilled: %d", loss)
    
    logging.debug("Pipe \n %s ", str(pipe))
    
    return inv, pipe, waste, loss, freshness, fulfilled, unfulfilled, d, flow

def update(n, m, A, lead, inv, pipe, prod, flow, fill, disp, source, sink):
    waste = 0
    
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
    #print "PIPE 1 sup", pipe[:,1]
    #print "PIPE 1 ret", pipe[:,2]

    pipe = np.roll(pipe, -1, axis=0) # aging
    pipe[-1:,:,:] = 0
    
    #print "aged pipe sup : ", pipe[:,1]
    #print "aged pipe ret", pipe[:,2]
    
    pipe = np.roll(pipe, -1, axis=2) # expiring
    pipe[:,:,-1:] = 0
    
    #print "expired pipe sup: ", pipe[:,1]
    #print "expired pipe ret", pipe[:,2]
    
    logging.debug("Pipeline update: aging\n %s \n", str(pipe))
    if np.min(pipe) < negTolerance:
        logging.warning("Pipe below 0 after pipe aging: \n %s", str(pipe))
    
    # Orders are transformed into flow vectors
    # 3a. inflow from upstream to pipeline inventory
    rows, cols = np.nonzero(A)
    for i in range(len(rows)):
        l = lead[rows[i],cols[i]]
        #print "L = ", l
        pipe[l-1][cols[i]] += np.sum(flow[:,cols[i],:],axis=0)
        #print "upstream sup ", i, " ", pipe[:,1]
        #print "upstream ret ", i, " ", pipe[:,2]
    logging.debug("Pipeline update: receiving from upstream\n %s \n",str(pipe))
    if np.min(pipe) < negTolerance:
        logging.warning("Pipe below 0 after receiving flow: \n %s", str(pipe))
    
    #print "uppstream pipe sup: ", pipe[:,1]
    #print "uppstream pipe ret: ", pipe[:,2]


    # 3b. outflow: shipment to downstream
    inv -= np.sum(flow,axis=1)
    #print "new inv after shipping sup", sum(inv[1])
    #print "new inv after shipping ret", sum(inv[2])
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
    
    #print "inv - fulfill sup", sum(inv[1])
    #print "inv - fulfill ret", sum(inv[2])
    
    # 6. disposal
    inv -= disp
    waste += sum2(disp)
    logging.debug("Disposal:\n %s \n", str(disp))
    logging.debug("Inventory update: disposal\n %s \n", str(inv))
    if np.min(inv) < negTolerance:
        logging.warning("Inv below 0 after disposal: \n %s", str(inv))
    
    # 7. on-hand inventory aging
    
    # Assumptions: the source nodes are just holders for inv, wastes there 
    # are not included. We only include wastes on things that non-source nodes
    # ordered.
    #waste += sum2(inv[:,-1:])
    for i in range(n):
        if source[i] == 0:
            waste += inv[i,0]
    inv = np.roll(inv, -1, axis=1)
    inv[:,-1:] = 0
    logging.debug("Inventory update: aging\n %s \n", str(inv))

    return inv, pipe, waste


def sum2(array):
    return sum(sum(array))

def sum3(array):
    return sum(sum(sum(array)))


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


    supplier = Supplier()
    supplier.run(n, m, A, lead, inv, pipe, dparams, yparams, source, sink)

if __name__ == "__main__":
    main()


    '''
    convert state space to 21 + 21 = 42
    order[i,j] = base_stock.- (inv + sum(pipe[:,j])) where. i = source, j = dest
    '''

