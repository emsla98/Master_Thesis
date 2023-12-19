# Import libraries:
import mikeio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import sys
import importlib
import pandas as pd

sys.path.append("../")
plt.style.use("seaborn-v0_8-whitegrid")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA
from sklearn.linear_model import LinearRegression
from pysindy import SINDy, STLSQ

from IPython.display import HTML
from tqdm import tqdm


# RMSE function:
def rmse(y_true, y_pred, axis=None, weights=None):

    if axis is None:
        ret = np.sqrt(np.average(((y_true - y_pred)**2).astype(float), weights=weights))

    if axis == 0:
        ret = np.sqrt(np.average(((y_true - y_pred)**2).astype(float), axis=0, weights=weights))

    if axis == 1:
        ret = np.sqrt(np.average(((y_true - y_pred)**2).astype(float), axis=1, weights=weights))
    return ret

# ME function:
def me(y_true, y_pred, axis=None, weights=None):

    if axis is None:
        ret = np.average(((y_true - y_pred)**2).astype(float), weights=weights)

    if axis == 0:
        ret = np.average(((y_true - y_pred)**2).astype(float), axis=0, weights=weights)
        
    if axis == 1:
        ret = np.average(((y_true - y_pred)**2).astype(float), axis=1, weights=weights)

    return ret

# MinAE function:
def minae(y_true, y_pred, axis=None, weights=None):

    if axis is None:
        ret = np.min(np.abs((y_true - y_pred)))

    if axis == 0:
        ret = np.min(np.abs((y_true - y_pred)), axis=0)

    if axis == 1:
        ret = np.min(np.abs((y_true - y_pred)), axis=1)

    return ret

# MAE function:
def mae(y_true, y_pred, axis=None, weights=None):

    if axis is None:
        ret = np.average(np.abs((y_true - y_pred)), weights=weights)

    if axis == 0:
        ret = np.average(np.abs((y_true - y_pred)), axis=0, weights=weights)

    if axis == 1:
        ret = np.average(np.abs((y_true - y_pred)), axis=1, weights=weights)

    return ret

# MaxAE function:
def maxae(y_true, y_pred, axis=None, weights=None):

    if axis is None:
        ret = np.max(np.abs((y_true - y_pred)))

    if axis == 0:
        ret = np.max(np.abs((y_true - y_pred)), axis=0)

    if axis == 1:
        ret = np.max(np.abs((y_true - y_pred)), axis=1)

    return ret


# Evaluate function:
def compute_metric(y_true_df, y_pred_dict, kwargs):
    """
    Function for computing a given metric between two datasets.

    ----------
    Parameters:
    -----------
    y_true_df : pd.DataFrame
        DataFrame of true values.
    y_pred_dict : dict
        Dictionary of predicted values.
    kwargs : dict
        Dictionary of keyword arguments:
            metric : "rmse", "me", "mae", "minae", "maxae"
                Metric to compute.
            axis : "time", "space"
                Axis to compute metric over.
            weights: 
                Array of mesh element areas. 
            neg: bool
                Whether to return negative metric.
    """

    # Unpack kwargs:
    metric = kwargs["metric"]
    axis = kwargs["axis"]
    weights = kwargs["weights"]
    neg = kwargs["neg"]

    # Check if metric is valid:
    if metric not in ["rmse", "me", "mae", "minae", "maxae"]:
        raise ValueError("Invalid metric. Options: 'rmse', 'me', 'mae', 'minae', 'maxae'.")
    
    # Check if axis is valid:
    if axis not in ["time", "space"]:
        raise ValueError("Invalid axis. Options: 'time', 'space'.")
    
    # Convert axis to int:
    if axis == "time":
        axis = 1
    
    if axis == "space":
        axis = 0

    # Convert y_true_df to dict:
    y_true_dict = {}

    for key in y_pred_dict.keys():
        
        # Filter data columns:
        data_tmp = y_true_df.filter(regex=f"^{key}")  # Filter data
        
        # Remove surplus rows: (Observations)
        data_tmp = data_tmp.loc[y_pred_dict[key].index] 
        y_true_dict[key] = data_tmp
    
    # Initialize error:
    error = []

    # Loop over all keys:
    for key in y_true_dict.keys():

        # Retrieve true and predicted values:
        y_true = y_true_dict[key]
        y_pred = y_pred_dict[key]
        

        if (y_true.isna().sum().sum()) > 0:
            print("NaNs present in y_true")
            
        if np.sum(y_pred.isna().sum().sum()) > 0:
            print("NaNs present in y_pred")
        
        # Compute metric:
        if metric == "rmse":
            ret = rmse(y_true, y_pred, axis=axis, weights=weights)
        
        if metric == "me":
            ret = me(y_true, y_pred, axis=axis, weights=weights)
        
        if metric == "mae":
            ret = mae(y_true, y_pred, axis=axis, weights=weights)
        
        if metric == "minae":
            ret = minae(y_true, y_pred, axis=axis, weights=weights)
        
        if metric == "maxae":
            ret = maxae(y_true, y_pred, axis=axis, weights=weights)

        if neg:
            ret = -ret

        # Add to error:
        error.append(ret.mean())

    # Return average error:
    return np.array(error).mean()


# Retrieve data format:
def get_mikeio_format():

    path = "../../Data/DHI_yr_sim/Area.dfsu"

    return mikeio.read(path, time=0, items=0)[0]

# Plot function:
def plot(
        data=None,          # Original data
        data_copy=None,     # Reconstructed data
        frame=0,            # Starting frame
        extra="Reconstructed with __" 
                            # Title of the plot
        ):

    cbar_min = np.min([data.Surface_elevation[frame,:].values.min(),
                    data_copy.Surface_elevation[frame,:].values.min()])
    cbar_max = np.max([data.Surface_elevation[frame,:].values.max(), 
                    data_copy.Surface_elevation[frame,:].values.max()])

    print("Cbar min: ", cbar_min)
    print("Cbar max: ", cbar_max)

    # Create figure and subplot for comparison:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot original data:
    data.Surface_elevation[frame,:].plot(ax=ax1, 
                                        vmin=cbar_min, vmax=cbar_max)
    ax1.set_title("Original data")

    # Plot reconstructed data:
    data_copy.Surface_elevation[frame,:].plot(ax=ax2,
                                                vmin=cbar_min, vmax=cbar_max)
    ax2.set_title("Reconstructed data")

    # RMSE:
    rmse_val = rmse(data.Surface_elevation[frame,:].values,
                    data_copy.Surface_elevation[frame,:].values)

    # Plot difference:
    (data.Surface_elevation[frame,:] - data_copy.Surface_elevation[frame,:]).plot(ax=ax3, #vmin=cbar_min, vmax=cbar_max, 
            cmap="seismic")
    ax3.set_title("Difference. RMSE: " + str(rmse_val))

    # Super title:
    plt.suptitle(f"Comparison of frame {frame}. "+ extra)

    # Show figure:
    plt.show()

    return

# Animation function:
def animate_plot(
        data=None,          # Original data 
        data_copy=None,     # Reconstructed data
        frame=0,            # Starting frame
        n_frames=1,         # Number of frames to animate           
        extra=""            # Title of the plot
        ):

    print(n_frames)

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(18, 6), ncols=3, )

    add_cbar = True

    cbar_min = np.min([data.Surface_elevation.values.min(),
                    data_copy.Surface_elevation.values.min()])
    cbar_max = np.max([data.Surface_elevation.values.max(), 
                    data_copy.Surface_elevation.values.max()])

    cbar_diff_min = (data.Surface_elevation.values - \
                        data_copy.Surface_elevation.values).min()
    cbar_diff_max = (data.Surface_elevation.values - \
                        data_copy.Surface_elevation.values).max()
    

    def update(*args):
        ax1.clear(); ax2.clear(); ax3.clear()

        global self, frame, add_cbar
        global cbar_min, cbar_max, cbar_diff_min, cbar_diff_max

        # Set super title:
        fig.suptitle(f"Comparison of frame {frame}. \
                    Reconstruction from"+extra)
        
        # Plot original data:
        data.Surface_elevation[frame,:].\
            plot(ax=ax1, vmin=cbar_min, vmax=cbar_max,
                add_colorbar=add_cbar)
        
        ax1.set_title("Original data")

        # Plot reconstructed data:
        data_copy.Surface_elevation[frame,:].\
            plot(ax=ax2, vmin=cbar_min, vmax=cbar_max,
                add_colorbar=add_cbar)
        ax2.set_title("Reconstructed data")

        rmse_val = rmse(data.Surface_elevation[frame,:],
                        data_copy.Surface_elevation[frame,:])

        # Plot difference:
        (data.Surface_elevation[frame,:] - \
        data_copy.Surface_elevation[frame,:]).\
            plot(ax=ax3, vmin=cbar_diff_min, vmax=cbar_diff_max, 
                cmap="seismic",
                add_colorbar=add_cbar)
        ax3.set_title(f"Difference. RMSE: {rmse_val:.5f}")

        # Increment frame:
        frame += 1

        # Disable colorbar after first frame:
        add_cbar = False

        return ax1, ax2, ax3,

    # Animate:
    ani = animation.FuncAnimation(fig, update, interval=500,
                                frames=tqdm(range(n_frames)))

    print(n_frames)

    plt.close()
    plt.show()

    return ani

# Animation function:
def animate_plot2(
        df_true=None,       # Original data 
        df_pred=None,       # Reconstructed data
        frame=0,            # Starting frame
        n_frames=1,         # Number of frames to animate           
        extra=""            # Title of the plot
        ):

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 8), ncols=3, )
    
    img_data_true = get_mikeio_format()
    img_data_pred = get_mikeio_format()
    img_data_diff = get_mikeio_format()
    
    add_cbar = True
    
    cbar_max = np.max(np.abs(df_true.values))
    cbar_min = -cbar_max
    
    def update(*args):
        
        ax1.clear(); ax2.clear(); ax3.clear()

        global self, frame, add_cbar
        global cbar_min, cbar_max, cbar_diff_min, cbar_diff_max

        # Set super title:
        fig.suptitle(f"Comparison of frame {frame}. \
                    Reconstruction from"+extra)
        
        # Plot original data:
        img_data_true.values[:] = df_true.values[frame, :]
        
        img_data_true.plot(ax=ax1, 
                      vmin=cbar_min, vmax=cbar_max, 
                      add_colorbar=add_cbar)
        
        ax1.set_title("Original data")

        # Plot reconstructed data:
        img_data_pred.values[:] = df_pred.values[frame, :]
        
        img_data_pred.plot(ax=ax2, 
                      vmin=cbar_min, vmax=cbar_max,
                      add_colorbar=add_cbar)
        
        ax2.set_title("Reconstructed data")

        rmse_val = rmse(img_data_true.values, img_data_pred.values)

        # Plot difference:
        img_data_diff.values[:] = df_true.values[frame, :] - df_pred.values[frame, :]
        
        img_data_diff.plot(ax=ax3, 
                      vmin=cbar_diff_min, vmax=cbar_diff_max, 
                      cmap="seismic", add_colorbar=add_cbar)
        
        ax3.set_title(f"Difference. RMSE: {rmse_val:.5f}")

        # Increment frame:
        frame += 1

        # Disable colorbar after first frame:
        add_cbar = False

        return ax1, ax2, ax3,

    # Animate:
    ani = animation.FuncAnimation(fig, update, interval=500,
                                frames=tqdm(range(n_frames)))

    plt.close()
    plt.show()

    return ani



# Temporal plot function:
def plot_temporal_samples(samples):
    """
    Function for plotting temporal distribution of samples.

    ----------
    Parameters:
    -----------
    samples : array-like
        Array of indices of samples in the dataset.
    """



    # Create subplot with 3 rows and 1 column:
    fig, axs = plt.subplots(4, 1, figsize=(12, 6))

    fig.suptitle("Distribution of samples")

    # Plot:
    axs[0].set_title("Hourly distribution")
    axs[0].hist(samples % 48, bins=48, align="left", rwidth=0.8, color="blue")
    axs[0].set_xlabel("Time of day [½ hours]")
    axs[0].set_xticks(np.arange(0, 48, 2))
    axs[0].set_xticklabels(np.arange(0, 25, 1))

    axs[1].set_title("Weekly distribution")
    axs[1].hist((samples // 48) % 7, bins=7, align="left", rwidth=0.8, color="blue",
                range=(0, 7))
    axs[1].set_xlabel("Day of week")
    axs[1].set_xticks(np.arange(0, 7, 1))
    axs[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    axs[2].set_title("Monthly distribution")
    axs[2].hist((samples // (48*7)) % 4, bins=4, align="left", rwidth=0.8, 
                color="blue",range=(0, 4))
    axs[2].set_xlabel("Week of month")
    axs[2].set_xticks(np.arange(0, 4, 1))
    axs[2].set_xticklabels(["1", "2", "3", "4"])


    axs[3].set_title("Yearly distribution")
    axs[3].hist((samples // (48*7*4)) % 12, 
                bins=12, align="left", rwidth=0.8, color="blue", range=(0, 12))
    axs[3].set_xlabel("Month of year")
    axs[3].set_xticks(np.arange(0, 12, 1))
    axs[3].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                            "Sep", "Oct", "Nov", "Dec"])

    plt.tight_layout()

    plt.show()

    return

# Equitemporal sampling function:
def equitemporal_sampling(
        max_mth=12, max_wk=4,
        max_day=7, max_hr=48,
        n_samples=1):
    """
    Function for creating equitemporal sampling of a given dataset.
    """
    mth = wk = day = hr = 0

    ids = []

    for i in range(n_samples):
        mth = (mth + 1) % max_mth
        wk = (wk + 1) % max_wk
        day = (day + 1) % max_day
        hr = (hr + 1) % max_hr

        ids.append(hr + 48 * (day + 7 * (wk + 4 * mth)))

    return np.array(ids)

    
    def pca_data(data):

        # Scale data:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Perform PCA:
        pca = PCA()
        pca.fit(data_scaled)

        return pca, scaler

    # Create empty lists:
    pcas = []
    scalers = []
    recon_errs = np.zeros((len(datasets), len(datasets)))

    # Compute pcas:
    for dataset in datasets:
        pca, scaler = pca_data(dataset)
        pcas.append(pca)
        scalers.append(scaler)
    
    # Compute reconstruction errors using different pca's:
    for i in range(len(pcas)):
        scaler = scalers[i]
        pca = pcas[i]

        for j in range(len(pcas)):
            data = datasets[j]
            data_scaled = scaler.transform(data)
            data_scaled_recon = pca.inverse_transform(pca.transform(data_scaled))
            data_recon = scaler.inverse_transform(data_scaled_recon)
            error = np.linalg.norm(data - data_recon)
            recon_errs[i, j] = error

    return pcas, scalers, recon_errs





