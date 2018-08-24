#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:34:02 2018

@author: pyzhang
"""
import numpy as np
import logging
#from scipy import stats
#import datetime

# Interface for DP and RL

# Model specifications:
#   A: adjancenty matrix for nodes (binary). A[i,j] = 1 if i feeds j, 0 otherwise.
#   lead: lead time matrix (integer). lead[i,j] is the lead time from i to j.
#   dparams: demand random variable parameters (Guassian). dparams[0] = mean
#           dparams[1] = standard deviation.
#   yparams: yield random variables (for farms). Gaussian, with mean yparams[0]
#       and standard deviation yparams[1]
#   n: (integer) number of nodes in the network
#   m: (integer) max shelf life
#   source: (binary) vector. source[i] = 1 if node i is farm, 0 otherwise.
#   sink: (binary) vector. sink[i] = 1 if node i is retailer facing customer 
#       demand. 0 otherwise.
#
# State:
#   x: inv state (float). x[i,l] is the amount of inventory at i with l periods
#       life remaining.
#   pipe: pipeline inv state (float). pipe[t,i,l] is the amount of inventory 
#           arriving to i in k periods, and currently have l periods of life
#
# Action:
#   orders: orders[i,j] is the amount to order from j to i in this period.
# 


# 1. Generate (random) demands and yields
# 2. Take actions (orders) as input, split orders to upstream, 
#      Decide the actual realized orders   
#
# Assumptions:
#   Nodes are strictly divided into levels. This simplifies the propagation 
#   of demand.
#   FIFO issuing except for source. Source sends freshest.
#   Production arrives at source nodes (push), other flows are pulled by orders
#   Each sink node faces the same demand (matters when setting base level 
#   for the non-sink and non-source nodes)
#    
# Input: n, A, 
#   lead[i,j]: lead time between i and j. Use 1000 to denote absence of link
#   inv[i,k]: amount of life-k inv for node i (on-hand)
#   pipe[l,i,k]: in-transit inventory, arriving in l periods
#   dparams: parameters to generate random demands [0] mean, [1] standard dev
#   yparams: parameters to generate random yields/capacities, [0] mean, [1] sd
#   sl: service level requirement
#   source: vector of 0 and 1 indicating whether a node is a farm (source)
#   sink: vector of 0 and 1 indicating whether a node is a retailer (sink)  
#    
# Return: 
#   Total waste in the network
#   Demand fulfillment vector in the network
#   Demand loss
#    
def pull(orders, n, m, A, lead, inv, pipe, dparams, yparams, source, sink):  
    waste = 0
    loss = 0
    d = np.zeros(n)
    fulfilled = np.zeros(n)
    unfulfilled = d
    
    # Initialize a random d
    d = np.maximum(0, np.random.normal(dparams[0], dparams[1], n))
    d = np.minimum(d, sink * (dparams[0]+dparams[1]*10))
    
    # print "demand: ", d
    logging.debug("Demand vector:\n %s \n", str(d))
    
    # Initialize yield vector
    prod = np.zeros((n, m))
    prod[:, m-1] = np.random.normal(yparams[0], yparams[1], n)
    prod[:, m-1] = np.minimum(prod[:, m-1], source * (yparams[0]+yparams[1]*10))
    
    logging.debug("Production vector:\n %s \n", str(prod))
    
    # Set base stock level based on service level, lead time, every node
    logging.debug("Lead time matrix:\n %s \n", str(lead))
    # min lead time for each node
    l = np.amin(lead, 0)
    
    # Calculate the minimum freshness a node i needs to send to downstream
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
                toSink[i] = min(toSink[i], toSink[j] + lead[i, j])
    
    # Converting orders[i,j] to flows[i,j,k]
    # Generate flow as much as possible, up to the upstream on-hand inv level
    # Assumptions: 
    #   FIFO, 
    #   Proportional fulfillment of orders
    #   Inv used only when there's enough life left to survive the lead time
    # flow[i,j,k]: amount of life-k flow from i to j
    flow = np.zeros((n, n, m))
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                if source[i] == 1:
                    flow[i, j, m-1] = orders[i, j]
                else:
                    # l = lead[i,j] + 1
                    l = lead[i,j] + toSink[j] + 1
                    lastK = l
                    for k in range(m):
                        if k >= l:
                            if sum(inv[i,l:k+1])<=sum(orders[i, :]) and sum(orders[i, :]) > 0:
                                flow[i,j,l:k+1] = 1.0 * inv[i, l:k+1] * orders[i, j] / sum(orders[i, :])
                                lastK += 1
                    if lastK < m and lastK >= l and sum(orders[i,:]) > 0:
                        flow[i,j,lastK] = 1.0 * (sum(orders[i, :]) - sum(inv[i,
                                                                         l:lastK])) * orders[i, j] / sum(orders[i, :])
    
    # Transform d to fill array -- the actual demand fulfilled
    # fill[i,k] amount of life-k inventory being used by i to fulfill demand
    fill = np.zeros((n, m))
    for i in range(n):
        lastK = 0
        for k in range(m):
            if sum(inv[i, :k+1]) <= d[i]:
                fill[i, :k+1] = inv[i, :k+1]
                lastK += 1
        if lastK < m and d[i] > 0:
            fill[i, lastK] = d[i] - sum(inv[i, :lastK])
    logging.debug("Demand fulfilled:\n %s \n", str(fill))

    # Disposal decision
    # No intentionally disposal for this basic pull
    disp = np.zeros((n,m))
    
    # Call update function and accumulate waste, etc.
    logging.debug("Calling update function. \n")
    inv, pipe, newWaste = update(n, m, A, lead, inv, pipe, prod, flow, fill, disp, source, sink)
    logging.debug("Finished update function. \n")
    
    waste += newWaste
    
    fulfilled = sum(fill.T).T
    unfulfilled = d - fulfilled
    loss = sum(unfulfilled)
    
    # print "fulfilled sum=", sum(fulfilled)
    if sum(fulfilled) < 0.0001: # numerical hack for == 0
        freshness = 0
    else:
        freshness = np.dot(np.sum(fill,axis=0), np.arange(m)+1) / m / sum(fulfilled)
        
    logging.debug("Total demand filled: %d", sum2(fill))
    logging.debug("Total demand unfilled: %d", loss)
    
    logging.debug("Pipe \n %s ", str(pipe))
    
    return inv, pipe, waste, loss, freshness, fulfilled, unfulfilled, d


# Update system state.
#
#
# Events order: 
# Initial code (July 8)
#   1. closest pipeline inventory arriving
#   2. pipeline inventory aging and moving closer
#   3. inflow from upstream to pipeline inventory
#   4. production arrival
#   5. outflow: demand fulfillment
#   6. outflow: shipment to downstream
#   7. disposal
#   8. on-hand inventory aging
# Updated sequence (July 11)
#   Following the sequence of events in Chen, Pang, and Pan (2014)
#   a. Pipeline inventory updating, expiring, arriving on hand, 
#   b. Order placement (up to b), propagation from downstream to upstream
#      this initiates flows from a supply node to the pipeline of downstream
#   d. Demand realization and fulfillment (by source nodes), production arrival
#   e. Oldest inventory expire, and on-hand inventory aging
#       
# Assumptions: 
#   Source nodes receives production right away. Therefore, source nodes don't
#       have pipeline inventory
#   inv and pipe are not negative after update -- this is guaranteed by input
#       this function is just for
#       updating, not for deciding the issuing policy etc.
#   length of 3rd dimension of inv = m, max life time
#   lead time on arc (i,j) is fixed
#   Waste does not include pipeline waste -- it's parent's responsibility
# Input:
#   n: total number of nodes in the network
#   m: max shelf life of product
#   A[i,j]: 1 if i can feed j, 0 otherwise
#   lead[i,j]: integer, lead time between i and j
#   inv[i,k]: amount of life-k inv for node i (on-hand)
#   pipe[l,i,k]: in-transit inventory, arriving in l periods
#   prod[i,k]: production of life-k inv at source node i 
#   flow[i,j,k]: flow quantity of life-k inv from i to j 
#   fill[i,k]: amount of life-k inv at node i that is shipped out
#   disp[i,k]: amount of life-k inv to dipose at node i
# Return:   
#           on-hand inventory, 
#           pipeline inventory
#           total waste (expiration+disposal)
#

def update(n, m, A, lead, inv, pipe, prod, flow, fill, disp, source, sink):
    waste = 0
    
    negTolerance = -0.001

    logging.debug("Inventory on hand:\n %s", str(inv))
    logging.debug("Pipeline inventory pipe[l,i,k]:\n %s \n", str(pipe))
    
    # 1. closest pipeline inventory arriving
    inv += pipe[0]

    logging.debug("Inventory update: received from pipeline\n %s \n", str(inv))
    if np.min(inv) < negTolerance:
        logging.warning("Inv below 0 after receiving pipe: \n %s", str(inv))
    
    # 2. pipeline inventory aging and moving closer

    # aging
    pipe = np.roll(pipe, -1, axis=0)
    pipe[-1:, :, :] = 0

    # expiring
    pipe = np.roll(pipe, -1, axis=2)
    pipe[:, :, -1:] = 0
    logging.debug("Pipeline update: aging\n %s \n", str(pipe))
    if np.min(pipe) < negTolerance:
        logging.warning("Pipe below 0 after pipe aging: \n %s", str(pipe))
    
    # Orders are transformed into flow vectors
    # 3a. inflow from upstream to pipeline inventory
    rows, cols = np.nonzero(A)
    for i in range(len(rows)):
        l = lead[rows[i], cols[i]]
        pipe[l-1][cols[i]] += np.sum(flow[:, cols[i], :], axis=0)
    logging.debug("Pipeline update: receiving from upstream\n %s \n", str(pipe))
    if np.min(pipe) < negTolerance:
        logging.warning("Pipe below 0 after receiving flow: \n %s", str(pipe))
    
    # 3b. outflow: shipment to downstream
    inv -= np.sum(flow, axis=1)
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
    waste += sum2(disp)
    logging.debug("Disposal:\n %s \n", str(disp))
    logging.debug("Inventory update: disposal\n %s \n", str(inv))
    if np.min(inv) < negTolerance:
        logging.warning("Inv below 0 after disposal: \n %s", str(inv))
    
    # 7. on-hand inventory aging
    
    # Assumptions: the source nodes are just holders for inv, wastes there 
    # are not included. We only include wastes on things that non-source nodes
    # ordered.
    # waste += sum2(inv[:,-1:])
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


# Main function

# 3 nodes in the system
n = 3
# shelf life
m = 7
A = np.zeros((n, n))
# farm feeds supplier
A[0, 1] = 1
# supplier feeds farm
A[1, 2] = 1
lead = np.zeros((n, n),  dtype=np.int16) + 1000
# lead time from farm to supplier
lead[0, 1] = 2
# lead time from supplier to retailer
lead[1, 2] = 2
maxLead = 2
source = np.zeros(n)
sink = np.zeros(n)
# farm
source[0] = 1
# retailer
sink[2] = 1
# demand distribution
# TODO: set std to 15
dparams = np.array([10, 5])
# yield distribution
yparams = np.array([100, 2])
inv = np.maximum(np.zeros((n, m)), 0)
pipe = np.maximum(np.zeros((maxLead, n, m)), 0)
orders = np.zeros((n, n))
orders[0, 1] = 1
orders[1, 2] = 1

# initialize farm inventory
for i in range(7):
    inv[0, i] = 1000

# initialize farm pipe
for j in range(7):
    # 1 day away
    pipe[0, 0, j] = 1000
for k in range(7):
    # 2 days away
    pipe[1, 0, k] = 1000
