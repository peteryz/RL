import pickle
import sys
from progressbar import ProgressBar
import interface
from interface import *
import numpy as np
from math import sqrt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter


def get_base_stock_level(A, dparams, lead, i, z=2):
    """
    :param i: current node with parent k
    :param A: Adjacency matrix for node
    :param dparams: dparams[0] is the mean of dparams[0] is the std dev.
    :param lead: lead time periods (days) from parent to child i
    :param z: number of std devs to consider
    :return: base stock level for node i
    """
    # Need to consider stock in-transit
    # From the below, between 7 and 47, take 10 base stock values (s) and for those 10 "s" what is the avg. reward
    # create plot of s vs avg reward
    for parent in range(0, int(A.shape[0])):
        if A[parent, i] == 1:
            return dparams[0] * (lead[parent, i] + 1) + (z * dparams[1]) * (sqrt(lead[parent, i] + 1))


# To create axis with negative values in the correct direction for plotting
def neg_tick(x, pos):
    return '%.1f' % (-x if x else 0)


def check_base_stock(run_option, reward_io, supplier_io, retailer_io):
    """
    :param run_option: 0 to generate new rewards and write to memory_file, 1 to read from memory_file
    :param reward_io: rewards file to write results to or read results from
    :param supplier_io: supplier file to write results to or read results from
    :param retailer_io: retailer file to write results to or read results from
    :return: None
    """
    combinations = []
    rewards = []
    steps = []
    supplier_base_stock = []
    retailer_base_stock = []
    # check if inventory falls below s and order up to s
    # use get_base_stock_level and try values around it
    x_supplier = np.arange(5, 80, 5)
    y_retailer = np.arange(25, 100, 5)
    x, y = np.meshgrid(x_supplier, y_retailer)

    # x coordinates for plotting s vs reward
    # generate combinations from grid
    for i in range(0, len(x_supplier)):
        for j in range(0, len(y_retailer)):
            combinations.append((x[i, j], y[i, j]))

    if run_option == 0:
        pbar = ProgressBar(maxval=len(combinations))
        avg_reward = 0
        for pair in pbar(combinations):
            s_sup, s_ret = pair[0], pair[1]
            # Calculate average reward
            cumulative_sum = 0
            for step in range(10000):
                # Sum inventory and incoming inventory for farm, supplier, and retailer
                stock_level = interface.inv.sum(axis=1) + interface.pipe.sum(axis=2)
                # Sum inventory across days
                stock_level = stock_level.sum(axis=0)
                # Supplier
                if stock_level[1] < s_sup:
                    # Send action to order from farm to supplier
                    difference = s_sup - stock_level[1]
                    orders[0, 1] = difference
                else:
                    orders[0, 1] = 0
                # Retailer
                if stock_level[2] < s_ret:
                    # Send action
                    difference = s_ret - stock_level[2]
                    orders[1, 2] = difference
                else:
                    orders[1, 2] = 0
                    # Receive updated states and rewards
                interface.inv, interface.pipe, waste, loss, freshness, fulfilled, unfulfilled, d = \
                    pull(orders, n, m, A, lead, interface.inv, interface.pipe, dparams, yparams, source, sink)
                reward = (fulfilled[2] * freshness) - (0.3 * waste + loss)
                cumulative_sum = reward + cumulative_sum
                # rewards.append(reward)
                steps.append(step)
                # Plot avg reward; avoid division by 0 error
                avg_reward = cumulative_sum / (step + 1)
            rewards.append(avg_reward)
            supplier_base_stock.append(s_sup)
            retailer_base_stock.append(s_ret)

        # rewards[i] is the result of combining supplier_base_stock[i] and retailer_base_stock[i]
        # Write lists to pickle file
        retailer_file = open('pickles/' + retailer_io, 'w')
        supplier_file = open('pickles/' + supplier_io, 'w')
        rewards_file = open('pickles/' + reward_io, 'w')
        pickle.dump(retailer_base_stock, retailer_file)
        pickle.dump(supplier_base_stock, supplier_file)
        pickle.dump(rewards, rewards_file)
        print("Generation and writing complete. ")
    elif run_option == 1:
        # Load objects from file
        # For configuration std = 15 files: python policyEval.py 1 rewards.txt supplier4.txt retailer4.txt
        # For configuration std = 5 files: python policyEval.py 1 rewards2.txt supplier2.txt retailer2.txt
        try:
            retailer_file = open('pickles/' + retailer_io, 'r')
            supplier_file = open('pickles/' + supplier_io, 'r')
            rewards_file = open('pickles/' + reward_io, 'r')
            retailer_base_stock = pickle.load(retailer_file)
            supplier_base_stock = pickle.load(supplier_file)
            rewards = pickle.load(rewards_file)
        except IOError:
            print("Cannot load rewards: Either retailer, supplier or rewards file not found")
            sys.exit()

    retailer_base_stock = np.reshape(retailer_base_stock, (len(y_retailer), len(y_retailer)))
    supplier_base_stock = np.reshape(supplier_base_stock, (len(x_supplier), len(x_supplier)))
    rewards = np.reshape(rewards, (len(x_supplier), len(x_supplier)))

    # Create plots
    print("Creating plot")
    formatter = FuncFormatter(neg_tick)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.zaxis.set_major_formatter(formatter)
    ax.set_xlabel('Retailer Base Stock Level')
    ax.set_ylabel('Supplier Base Stock Level')
    ax.set_zlabel('Rewards')
    fig.suptitle('Figure showing Base Stock Levels vs Reward', fontsize=14, fontweight='bold')
    ax.plot_wireframe(retailer_base_stock, supplier_base_stock, (rewards * -1), rstride=1, cstride=1)
    # ax.scatter(retailer_base_stock, supplier_base_stock, [-v for v in rewards])
    # plt.bar(range(0, 1000), [-v for v in rewards])
    plt.show()


def policy_eval():
    gamma = 0.95
    theta = 0.00001
    # initialize V(s) for all s
    pass


def policy_improvement():
    pass


def collapse_inv():
    # sum across rows for inv / x
    no_rows = np.sum(interface.inv, 1).shape[0]
    print(interface.inv)
    print(no_rows)
    return np.reshape(np.sum(interface.inv, 1), (no_rows, 1))


# print(get_base_stock_level(interface.A, interface.dparams, interface.lead, 1))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Incorrect number or arguments"
        print "Usage: python policyEval.py <RUN CHOICE (0/1 to write/read)> " \
              "<rewards file> <supplier file> <retailer file>"
        sys.exit()
    if int(sys.argv[1]) == 0:
        check_base_stock(int(0), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]))
    else:
        check_base_stock(int(1), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]))

