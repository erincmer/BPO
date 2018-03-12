import numpy as np
import matplotlib
import scipy.interpolate

matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import math
from baselines.bench.monitor import load_results

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def ts2xy(ts, xaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
        y = ts.r.values
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
        y = ts.r.values
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
        y = ts.r.values
    else:
        raise NotImplementedError
    return x, y
def roundup(x):
    return int(math.ceil(x / 10.0)) * 10
def plot_curves(xy_list, xaxis, title):
    fig = plt.figure(figsize=(8,8))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    runs = 10

    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i//runs]

        if i % runs ==0:
            y_list = []
            x_list = []
            if i//runs == 0:
                label = "Original Loss"

            if i//runs == 1:

                label = "log(2-gratio)"

        # print(x.shape)
        # print(y.shape)
        # input("wait 1 ")
        # plt.scatter(x, y, s=2)
        x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
        # plt.plot(x, y_mean, color=color)
        # print(x.shape)
        y_list.append(y_mean)
        x_list.append(x)
        # input("wait 2 ")

        if len(y_list)==runs:
            shortest_length = min(map(len, y_list))
            print(list(map(len, y_list)))

            lst2 = [item[0] for item in x_list]
            m0 = max(lst2)
            lst2 = [item[-1] for item in x_list]
            m1 = min(lst2)

            xx = np.linspace(roundup(m0), roundup(m1-10), (m1 - m0)//10)
            y_list2 = []
            for e1,e2 in zip(x_list,y_list):
                y_interp = scipy.interpolate.interp1d(e1, e2)
                y_list2.append(y_interp(xx))

            y_list = y_list2
            y_all = np.array(y_list)
            y_min = np.min(y_all, axis=0)
            y_max = np.max(y_all, axis=0)
            y_mean = np.mean(y_all, axis=0)
            # plt.plot(x, y_mean, color=color, label=label)
            plt.plot(xx, y_mean, color=color, label=label)
            # print(y_interp(1839.0))
            # print(y_interp(1840.0))
            # print(y_interp(1841.0))
            # print(y_interp(1842.0))
            plt.legend(loc=4)
            plt.fill_between(xx, y_min,y_max , color=color,alpha=.3,lw=0.0, edgecolor="None")

    plt.grid()
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()
    fig.savefig('./Graphs/'+ title +'.pdf')  # save the figure to file
    plt.close(fig)
def plot_results(dirs, num_timesteps, xaxis, task_name):
    tslist = []
    for dir in dirs:
        ts = load_results(dir)

        ts = ts[ts.l.cumsum() <= num_timesteps]
        tslist.append(ts)
    xy_list = [ts2xy(ts, xaxis) for ts in tslist]
    plot_curves(xy_list, xaxis, task_name)

# Example usage in jupyter-notebook
# from baselines import log_viewer
# %matplotlib inline
# log_viewer.plot_results(["./log"], 10e6, log_viewer.X_TIMESTEPS, "Breakout")
# Here ./log is a directory containing the monitor.csv files

def main():
    import argparse
    import os

    env = "HalfCheetah-v2"
    alg = "trpo_mpi"
    res_list = []
    for loss in range(2):
        for trial in range(10):

                res_list.append("./"+alg+"/log/"+env+"/"+"Loss_"+str(loss)+"_Run_"+str(trial))

    print(res_list)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirs', help='List of log directories', nargs = '*', default=res_list)

    parser.add_argument('--num_timesteps', type=int, default=int(40e6))
    parser.add_argument('--xaxis', help = 'Varible on X-axis', default = X_TIMESTEPS)
    parser.add_argument('--task_name', help = 'Title of plot', default = env + "__" + alg)
    args = parser.parse_args()
    args.dirs = [os.path.abspath(dir) for dir in args.dirs]
    plot_results(args.dirs, args.num_timesteps, args.xaxis, args.task_name)
    plt.show()

if __name__ == '__main__':
    main()