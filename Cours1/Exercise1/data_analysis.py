# 
import numpy as np
import matplotlib.pyplot as plt

# 
#################### Data Analysis Functions  ########################


####### Question 2)
def get_average_spike_rate(spike_train):
    """ Function that estimates de mean spike rate 
    Args:
        spike train (numpy array)
    Return:
        mean spike rate (double)
    """
    if len(spike_train) == 0:
        print("Warning! The spike train has no datas!")
        return 0
    else:
        return np.sum(spike_train)/len(spike_train)*1000

####### Question 3)
def get_isi(spike_train):
    """ Function that calculates the difference between spike times
    Args:
        spike train (numpy array)
    Return:
            [histogram values and bin edges]
    """
    if len(spike_train) == 0:
        print("Warning! The spike train has no datas!")
        return 0
    ISI = []
    count = 0
    for i in range(len(spike_train)):
        if spike_train[i]==1:
            ISI.append(count)
            count = 0
        else:
            count += 1
    hist, bins = np.histogram(ISI, bins = 20)
    return hist, bins
    
####### Question 4)
def get_autocorrelation(spike_train, max_lag):
    """ Function that estimates de autocorrelation fro 0 to a given lag
    Args:
        spike train (numpy array)
        max lag (int)
    Return:
        autocorrelation (numpy array)
    """
    if len(spike_train) == 0:
        print("Warning! The spike train has no datas!")
        return 0
    autocorrelation = np.zeros(max_lag)
    for tau in range(max_lag):
        for t in range(len(spike_train) - max_lag):
            autocorrelation[max_lag - tau - 1] += spike_train[t]*spike_train[t + tau]
        autocorrelation[max_lag - tau - 1] = autocorrelation[max_lag - tau - 1]/(len(spike_train)-max_lag)*1000
    return autocorrelation


####### Question 5)
def detect_burst(spike_train, min_interval):
    """ Function that detects bursts of activity
    Args:
        spike train (numpy array)
        min_interval (int)
    Return:
        burst_interval (list)
    """
    burst_interval = []
    if len(spike_train) == 0:
        print("Warning! The spike train has no datas!")
        return 0
    for i in range(int(2*len(spike_train)/min_interval)-1):
        start = int(i * min_interval / 2)
        end = int((i+2) * min_interval / 2)
        if np.sum(spike_train[start:end]) > 0.002 * min_interval: # it is judged as an interval of bursts of activity
            if len(burst_interval) != 0 and start <= burst_interval[-1][1]:
                start = burst_interval[-1][0]
                burst_interval = burst_interval[:-1]
            burst_interval.append([start, end])
    return burst_interval


# %%
######################################################

# %%
def plot_spike_train(spike_train): #feel free to polish this function
    """ Function that plots the spike_train
    Args:
        spike train (numpy array)
        exercise_folder_path (string): complete path to your folder
        fig_name (string): figure name. Should end with an image extension, typically .png or .svg
    """
    if np.sum(spike_train) == 0:
        print("Warning! The spike train has zero spikes!")
    fig = plt.figure(figsize = [6,4])
    ax = fig.add_axes((0.1, 0.12, 0.8, 0.8))
    spike_times = np.argwhere(spike_train)
    for spike in spike_times:
        plt.axvline(x = spike)
    plt.xlim([0,spike_train.size])
    plt.ylim([0,1])
    plt.yticks([])
    plt.xlabel('Time (ms)')
    #plt.savefig(exercise_folder_path+fig_name)
    
    

# %%
def plot_autocorrelation(norm_autocorrelation,lag):
    fig = plt.figure()
    plt.plot(norm_autocorrelation)
    plt.xlabel('Time lag')
    plt.ylabel('Autocorrelation')
    locs, labels = plt.xticks()
    plt.xticks(locs[1:-1],(locs[1:-1]-lag).astype('int'))

# %%
def plot_isi(hist_vals, bin_edges):
    plt.figure()
    plt.bar(bin_edges[:-1], hist_vals,align = 'edge',width=(bin_edges[1]-bin_edges[0]))
    plt.xlabel('Time difference between consecutive spikes')
    plt.ylabel('Probability')
    plt.ylim([0,1.2*np.max(hist_vals)])
    
