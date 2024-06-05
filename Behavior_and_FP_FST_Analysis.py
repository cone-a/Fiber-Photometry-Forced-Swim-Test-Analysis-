# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:51:26 2024

@author: aaron.cone
"""



import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import numpy as np
import os
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



### Smoothes data

 ### Smoothes signal

def smooth_signal(x,window_len=10,window='flat'): 
    
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    The code taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                'flat' window will produce a moving average smoothing.

    output:
        the smoothed signal        
    """
    
    import numpy as np

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': # Moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[(int(window_len/2)-1):-int(window_len/2)]


#%%
# Fits exponential curve

import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit

# Fits exponential curve


def func(x, a, b, c): 
    return a * np.exp(-b * x) + c

#%%  



### BASELINE DATA

folder_path = 'path_to_photometry_files'

dfs_iso= {}
dfs_gcamp = {}

for file_name in os.listdir(folder_path):
    if file_name.endswith('_415_Signal.CSV' or '_415_Signal.csv') or file_name.endswith('_470_Signal.CSV' or '_470_Signal.csv'):
        # Extract the key as everything before the 3rd underscore
        key_parts = file_name.split('_')
        if len(key_parts) >= 3:
            key = '_'.join(key_parts[3:7])
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('_415_Signal.CSV' or '_415_Signal.csv'):
                dfs_iso[key] = pd.read_csv(file_path)
            elif file_name.endswith('_470_Signal.CSV' or '_470_Signal.csv'):
                dfs_gcamp[key] = pd.read_csv(file_path)

 
columns_to_rename = {'Region0G': 'Region', 'Region1G': 'Region', 'Region2G': 'Region', 'Region3G': 'Region'}

# Update column names in dfs_iso_baseline dictionary
for key, df in dfs_iso.items():
    dfs_iso[key] = df.rename(columns=columns_to_rename)

# Update column names in dfs_gcamp_baseline dictionary
for key, df in dfs_gcamp.items():
    dfs_gcamp[key] = df.rename(columns=columns_to_rename)
  

#%%  
i = 0

#### Raw signals

# Creates figure isobestic and gcamp signal
for iso, gcamp in zip(dfs_iso, dfs_gcamp):
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(211)
    ax1.plot(dfs_iso[iso]['Timestamp'], dfs_iso[iso]['Region'], 'blue', linewidth=1.5)
    ax1.title.set_text('Raw Isobestic')
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format x-axis labels to show time
    # ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))      ### Hour Interval
    ax2 = fig.add_subplot(212)
    ax2.plot(dfs_gcamp[gcamp]['Timestamp'], dfs_gcamp[gcamp]['Region'], 'purple', linewidth=1.5)
    ax2.title.set_text('Raw Gcamp')
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format x-axis labels to show time
    # ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3)) ### Hour Interval
    fig.suptitle(list(dfs_gcamp.keys())[i])
    i += 1


#%%

import matplotlib.dates as mdates

# Choose window you would like to smooth
smooth_win = 10
i = 0

# Creates figure isobestic and gcamp signal
for (key_iso, value_iso), (key_gcamp, value_gcamp) in zip(dfs_iso.items(), dfs_gcamp.items()):
    value_iso['smooth_isobestic'] = smooth_signal(value_iso['Region'], smooth_win)
    value_gcamp['smooth_gcamp_signal'] = smooth_signal(value_gcamp['Region'], smooth_win)
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(211)
    ax1.plot(value_iso['Timestamp'], value_iso['smooth_isobestic'], 'blue', linewidth=1.5)
    ax1.title.set_text('Smooth Isobestic')
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format x-axis labels to show time
    # ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))      ### Hour Interval
    ax2 = fig.add_subplot(212)
    ax2.plot(value_gcamp['Timestamp'], value_gcamp['smooth_gcamp_signal'], 'purple', linewidth=1.5)
    ax2.title.set_text('Smooth Gcamp')
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format x-axis labels to show time
    # ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3)) ### Hour Interval
    fig.suptitle(list(dfs_gcamp.keys())[i])
    i += 1




#%% 

# Calculates exponential curve for gcamp and isobestic signal

i = 0

for (key, value), (key2, value2) in zip(dfs_iso.items(),dfs_gcamp.items()):
    xvalue = np.linspace(0, len(value['smooth_isobestic']),len(value2['smooth_gcamp_signal']))
    popt_iso, pcov = curve_fit(func,xvalue,value['smooth_isobestic'], maxfev=10000)
    popt_gcamp, pcov = curve_fit(func,xvalue,value2['smooth_gcamp_signal'], maxfev=10000)
    value['iso_popt'] = func(xvalue, *popt_iso)
    value2['gcamp_popt'] = func(xvalue, *popt_gcamp)
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(211)
    ax1.plot(value['Timestamp'], value['smooth_isobestic'],'red',linewidth=1.5)
    ax1.plot(value['Timestamp'], func(xvalue, *popt_iso),'blue',linewidth=1.5, label='iso')
    ax1.set_title('Isobestic Signal')
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format x-axis labels to show time
    ax2 = fig.add_subplot(212)
    ax2.plot(value2['Timestamp'], value2['smooth_gcamp_signal'],'red',linewidth=1.5)
    ax2.plot(value2['Timestamp'], func(xvalue, *popt_gcamp), 'blue',linewidth=1.5, label='gcamp')
    ax2.set_title('Gcamp Signal')
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format x-axis labels to show time
    fig.suptitle(list(dfs_gcamp.keys())[i])
    i += 1


#%% 

from sklearn.linear_model import Lasso

# Starts to label each mouse/figure at position 0 
i = 0 

for (key, value), (key2, value2) in zip(dfs_iso.items(),dfs_gcamp.items()):
    reference = (value['smooth_isobestic'] - value['iso_popt']) ## removes baseline from isobestic signal
    signal = (value2['smooth_gcamp_signal'] - value2['gcamp_popt']) ## removes baseline from gcamp signal
    z_reference = np.array((reference - np.median(reference)) / np.std(reference)) ## standardize isobestic signal
    z_signal = np.array((signal - np.median(signal)) / np.std(signal)) ## standardize gcamp signal
    lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000, positive=True, 
                random_state=9999, selection='random')
    n = len(z_reference)
    lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))
    z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)  ## Aligns isobestic signal to gcamp signal
    value2['zdFF'] = (z_signal - z_reference_fitted) ## subtracted, zscored, and fitted signal
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(value2['Timestamp'], value2['zdFF'], 'black')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('zdFF')
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format x-axis labels to show time
    fig.suptitle(list(dfs_gcamp.keys())[i])
    i += 1




#%%

# Load in behavior files for animals from LabGym


### BASELINE DATA

folder_path = 'path_to_behavior_files'

dfs_behav = {}

for file_name in os.listdir(folder_path):
    if file_name.endswith('_probability.CSV') or file_name.endswith('_probability.csv'):
        # Extract the key as everything before the 4th underscore
        key_parts = file_name.split('_')
        if len(key_parts) >= 4:
            key = '_'.join(key_parts[0:4])
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path).T # Transpose the DataFrame
            df = df.drop(df.index[0]) # Fix the index
            df = df[0].str.split(',', expand=True)
            df = df.applymap(lambda x: x.strip("[]'\"")) # Remove brackets and quotes
            df = pd.DataFrame(df)
            df = df.reset_index(level=0)
            dfs_behav[key] = df
            
#%%

### All NAs to beginning to make all data the same

# Add NA values and 'Time' column to each DataFrame
for key, behavior in dfs_behav.items():
    if key == 'C76':  # Specific case for C76 animals
        n = 1200
    else:  # Default case
        n = 150
        
    
    add_na = pd.DataFrame(['NA'] * n)
    behavior = pd.DataFrame(pd.concat([add_na, behavior], ignore_index=True))
    behavior['Time'] = pd.Series(np.arange(start=0.0, stop=len(behavior) * 0.033, step=0.033))
    
    # Update the dictionary with the modified DataFrame
    dfs_behav[key] = behavior

#%%



# Merge zdFF_data with behavior DataFrame and rename columns
for mouse in dfs_gcamp:
    zdFF_data = pd.DataFrame(dfs_gcamp[mouse]['zdFF'])
    behavior = dfs_behav.get(mouse)

    if behavior is not None:
        zdFF_behav = pd.merge(behavior, zdFF_data, left_index=True, right_index=True)
        zdFF_behav = zdFF_behav.drop(columns=[1, 'index'], errors='ignore')  # Ensure 'index' column is dropped
        zdFF_behav.columns = ['behavior', 'Time', 'zscore'] if len(zdFF_behav.columns) == 3 else zdFF_behav.columns  # Rename columns only if there are exactly 3
        dfs_behav[mouse] = zdFF_behav


#%%


i = 0

# Initiate figure



for mouse in dfs_behav:

    # Colors of behaviors
    palette = {
        'NA': '#000000', # black
        'swimming': '#FFA500', # orange
        'hindpaw swimming': '#00D100', # green
        'climbing': '#ff0000', # red
        'floating': '#00008B'} # blue
    
    
    fig, ax = plt.subplots(figsize = (25,5))
    
    
    sns.scatterplot(data = dfs_behav[mouse],x = 'Time', y = 'zscore', hue = 'behavior', palette = palette, s=10)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.xticks(rotation=45)
    
    ax.margins(x=0)
    
    fig.suptitle("Baseline:" + list(dfs_behav.keys())[i])
    i += 1





#%%

i = 0
lets_see_dict = {}

for mouse in dfs_behav:
    behavior_df = dfs_behav[mouse]
    
    # Map behavior events to binary values
    behavior_df['event'] = behavior_df['behavior'].map({'floating': 0, 'swimming': 1, 'hindpaw swimming': 1, 'climbing': 1})
    # behavior_df['event'] = behavior_df['behavior'].map({'floating': 1, 'swimming': 0, 'hindpaw swimming': 0, 'climbing': 0}) ### FOR IMMOBILITY ANALYSIS
    
    # Smooth the spike data
    behavior_df["event_smooth"] = behavior_df["event"].groupby(np.arange(len(behavior_df)) // 30).transform(lambda x: x.mode()[0] if not x.isnull().all() else np.nan)
    


    # Define the pattern of zeros and ones
    n_zeros = 150
    n_ones = 150
    zeros = [0] * n_zeros
    ones = [1] * n_ones
    zero_one = zeros + ones
    
    overlap = behavior_df['event_smooth'].fillna(0).astype(int).values.tolist()
    
    # Find where the pattern matches in the data
    hmmm = [(i, i + len(zero_one)) for i in range(len(overlap) - len(zero_one) + 1) if overlap[i:i + len(zero_one)] == zero_one]
    
    arr = np.array(hmmm)
    arr_list = [row[0] for row in arr]
    arr_list = [x + n_zeros for x in arr_list]  # Time point of change from 0 to 1
    
    lets_see = behavior_df['Time'].iloc[arr_list].values.tolist()
    
    # Save the lets_see list for the current mouse
    lets_see_dict[mouse] = lets_see
    
    fig, ax = plt.subplots(figsize=(25, 5))

    
    sns.scatterplot(data=behavior_df, x='Time', y='zscore', hue=  'event_smooth', ax=ax)
    
    # Add vertical lines at the specific time points
    for x in lets_see:
        ax.axvline(x=x, ymin=-2, ymax=4, linestyle='dashed', color='black')
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.xticks(rotation=45)
    
    # Annotate the vertical lines with their indices
    for idx, x in enumerate(lets_see):
        ax.text(x, 4, f"{idx}", verticalalignment='center')

    ax.margins(x=0)
    
    fig.suptitle("Behavior Aligned Photometry Events:" + mouse)
    i += 1

# Show the plot
plt.show()

#%%

# # ### FOR PERI-EVENT ANALYSIS ###

# Peri-event histogram for continuous values.
def contvar_peh(var_ts, var_vals, ref_ts, min_max, bin_width=False):
    if bin_width:
        ds_ts = np.linspace(var_ts.min(), var_ts.max(), int((var_ts.max() - var_ts.min()) / bin_width))
        ds_vals = np.interp(ds_ts, var_ts, var_vals)
        rate = bin_width
    else:
        rate = np.diff(var_ts).mean()
        ds_ts, ds_vals = (np.array(var_ts), np.array(var_vals))
        
    left_idx = int(min_max[0] / rate)
    right_idx = int(min_max[1] / rate)
    
    all_idx = np.searchsorted(ds_ts, ref_ts, "right")
    all_trials = []
    for idx in all_idx:
        if idx + left_idx >= 0 and idx + right_idx <= len(ds_vals):
            all_trials.append(ds_vals[idx + left_idx:idx + right_idx])
    
    return np.vstack(all_trials) if all_trials else np.array([])




to_start = -10
to_end = 10

zdFF_results = {}


# Process each mouse using specific lets_see time points
for mouse, lets_see in lets_see_dict.items():
    if mouse == 'Vglut_flp_C70_F3':
        ref_ts = lets_see[2:9]  # Choose specific time points for mouse 70F3
    elif mouse == 'Vglut_flp_C69_M0':
        ref_ts = lets_see
    elif mouse == 'Vglut_flp_C66_F3':
        ref_ts = lets_see[2:6]
    elif mouse == 'Vglut_flp_C72_M2':
        ref_ts = [230.175, 275.715, 295.515, 395.505, 619.245, 706.365]
    elif mouse == 'Vglut_flp_C71_F2':
        ref_ts = lets_see
    elif mouse == 'Vglut_flp_C69_M2':
        ref_ts = [307.395, 783.585, 843.975]
    elif mouse == 'Vglut_flp_C76_M0':
        ref_ts = [380, 475, 510, 530, 545]
    elif mouse == 'Vglut_flp_C76_M1':
        ref_ts = [243.045, 309.375, 508.365, 609.345, 699.435, 764.775]
    elif mouse == 'Vglut_flp_C76_M3':
        ref_ts = [279.675, 395, 520, 553, 630, 800]
    else:
        ref_ts = lets_see  # Default to original lets_see for other mice
    
    # Call contvar_peh with the specific reference time points for each mouse
    trials = contvar_peh(
        var_ts=dfs_behav[mouse]['Time'],
        var_vals=dfs_behav[mouse]['zscore'],
        ref_ts=ref_ts,
        min_max=(to_start, to_end),
        bin_width=False)
    
    if len(trials) > 0:
        zdFF_results[mouse] = trials
    else:
        print(f"No valid trials found for mouse {mouse}")





#%%
import pandas as pd
from itertools import chain

# Formatted so can process SEM and heatmap
to_line_points = chain.from_iterable(zdFF_results.values())
lined_up_points = np.array(list(to_line_points))

# creates time series for plotting, add a zero after events (e.g. num = events[0}.size])
# time_peri_event = np.linspace(start = to_start, stop = to_end, num = events[1].size, retstep=0.25)
time_peri_event = np.linspace(start = to_start, stop = to_end, num = lined_up_points[0].size, retstep=0.25)



#%%


# Calculates means for all data points
points = lined_up_points.mean(axis=0)

# Calculates standard error of mean for data points
points_sem = stats.sem(lined_up_points)

# Creates dataframe to plot 
to_plot = pd.DataFrame({'Time': time_peri_event[0], 'zdFF': points})



mean_neg = to_plot[to_plot['Time'] < 0]['zdFF'].mean()

# Subtract the mean from all 'zdFF' values
to_plot['zdFF_normalized'] = to_plot['zdFF'] - mean_neg


#%%

# Make line plot figure
fig, ax = plt.subplots(figsize=(16, 10)) # you change dimensions of plot here Width x Length

# ax = plt.gca() # needed for line below - change y axis min and max
# ax.set_ylim([-1.5, 2.5]) #change y axis min and max

# Makes line plot
ax.plot('Time', 'zdFF_normalized', data = to_plot)
ax.fill_between(to_plot['Time'], to_plot['zdFF_normalized'] - points_sem, 
                    to_plot['zdFF_normalized'] + points_sem, alpha=0.15)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Z-Score')
ax.set_title('Aggregrate Behavior Change Peri-Event')
ax.margins(x=0)    
        


# Create heatmap

# Concatenate all data
combined_data = []

for mouse, data in zdFF_results.items():
    if data.size == 0:
        print(f"No data to plot for mouse {mouse}")
        continue
    combined_data.append(data)

# Check if combined_data is not empty
if combined_data:
    # Concatenate all data along the vertical axis (axis=0)
    combined_data = np.vstack(combined_data)

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(combined_data, cbar=True, ax=ax, xticklabels=False, yticklabels=False)
    
    ax.set_title("Combined Heatmap of zdFF peri-event activity for all mice")
    ax.set_xlabel("Time bins")
    ax.set_ylabel("Event trials")
    
    plt.show()
else:
    print("No data available to plot.")




#%%


##### RANDOM TRIALS ### 

### 5 per animal

zdFF_results_random = {}

# Process each mouse to generate peri-event histograms
for mouse, behavior_df in dfs_behav.items():
    ref_ts = np.random.choice(behavior_df['Time'], 5)
    zdFF_results_random[mouse] = contvar_peh(
        var_ts=behavior_df['Time'],
        var_vals=behavior_df['zscore'],
        ref_ts=ref_ts,
        min_max=(to_start, to_end),
        bin_width=False
    )




#%%

##### RANDOM TRIALS ####

import pandas as pd
from itertools import chain

# Formatted so can process SEM and heatmap
to_line_points = chain.from_iterable(zdFF_results_random.values())
lined_up_points = np.array(list(to_line_points))

# creates time series for plotting, add a zero after events (e.g. num = events[0}.size])
# time_peri_event = np.linspace(start = to_start, stop = to_end, num = events[1].size, retstep=0.25)
time_peri_event = np.linspace(start = to_start, stop = to_end, num = lined_up_points[0].size, retstep=0.25)



#%%


# Calculates means for all data points
points = lined_up_points.mean(axis=0)

# Calculates standard error of mean for data points
points_sem = stats.sem(lined_up_points)

# Creates dataframe to plot 
to_plot = pd.DataFrame({'Time': time_peri_event[0], 'zdFF': points})



mean_neg = to_plot[to_plot['Time'] < 0]['zdFF'].mean()

# Subtract the mean from all 'zdFF' values
to_plot['zdFF_normalized'] = to_plot['zdFF'] - mean_neg


#%%

# Make line plot figure
fig, ax = plt.subplots(figsize=(16, 10)) # you change dimensions of plot here Width x Length

# ax = plt.gca() # needed for line below - change y axis min and max
# ax.set_ylim([-1.5, 2.5]) #change y axis min and max

# Makes line plot
ax.plot('Time', 'zdFF_normalized', data = to_plot)
ax.fill_between(to_plot['Time'], to_plot['zdFF_normalized'] - points_sem, 
                    to_plot['zdFF_normalized'] + points_sem, alpha=0.15)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Z-Score')
ax.set_title('Aggregrate RANDOM Behavior Change Peri-Event')
ax.margins(x=0)    
        


# Create heatmap

# Concatenate all data
combined_data = []

for mouse, data in zdFF_results.items():
    if data.size == 0:
        print(f"No data to plot for mouse {mouse}")
        continue
    combined_data.append(data)

# Check if combined_data is not empty
if combined_data:
    # Concatenate all data along the vertical axis (axis=0)
    combined_data = np.vstack(combined_data)

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(combined_data, cbar=True, ax=ax, xticklabels=False, yticklabels=False)
    
    ax.set_title("Combined RANDOM Heatmap of zdFF peri-event activity for all mice")
    ax.set_xlabel("Time bins")
    ax.set_ylabel("Event trials")
    
    plt.show()
else:
    print("No data available to plot.")
