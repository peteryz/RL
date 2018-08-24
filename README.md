# blockchain-interface. Interface for DP and RL

# Model specifications:
A: adjancenty matrix for nodes (binary). A[i,j] = 1 if i feeds j, 0 otherwise.

   lead: lead time matrix (integer). lead[i,j] is the lead time from i to j.
   
   dparams: demand random variable parameters (Guassian). dparams[0] = mean
           dparams[1] = standard deviation.
   
   yparams: yield random variables (for farms). Gaussian, with mean yparams[0]
       and standard deviation yparams[1]
   
   n: (integer) number of nodes in the network
   
   m: (integer) max shelf life
   
   source: (binary) vector. source[i] = 1 if node i is farm, 0 otherwise.
   
   sink: (binary) vector. sink[i] = 1 if node i is retailer facing customer 
       demand. 0 otherwise.

# State:
x: inv state (float). x[i,l] is the amount of inventory at i with l periods
       life remaining.
   
   pipe: pipeline inv state (float). pipe[t,i,l] is the amount of inventory 
           arriving to i in k periods, and currently have l periods of life

# Action:
   orders: orders[i,j] is the amount to order from j to i in this period.
 
