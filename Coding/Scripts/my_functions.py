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
def rmse(y_true, y_pred, axis=None):

    if axis is None:
        ret = np.sqrt(np.mean((y_true - y_pred)**2))

    if axis == 0:
        ret = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))

    if axis == 1:
        ret = np.sqrt(np.mean((y_true - y_pred)**2, axis=1))

    return ret

# MAPE function:
def mae(y_true, y_pred, axis=None):

    if axis is None:
        ret = np.mean(np.abs((y_true - y_pred)))

    if axis == 0:
        ret = np.mean(np.abs((y_true - y_pred)), axis=0)

    if axis == 1:
        ret = np.mean(np.abs((y_true - y_pred)), axis=1)

    return ret

# MAPE function:
def mape(y_true, y_pred, axis=None):

    if axis is None:
        ret = np.mean(np.abs((y_true - y_pred) / y_true))

    if axis == 0:
        ret = np.mean(np.abs((y_true - y_pred) / y_true), axis=0)

    if axis == 1:
        ret = np.mean(np.abs((y_true - y_pred) / y_true), axis=1)

    return ret


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

# Super class for models:
class myModels():
    
    def __init__(self):
        pass
    
    # Templates:
    class ModelTemplate:
        """
        Ensures that the methods fit, predict, and fit_predict are 
        instantiated for all classes based on this template.
        """

        def fit(self, X, y=None):
            raise NotImplementedError("fit method must be implemented in subclass")

        def predict(self, X):
            raise NotImplementedError("predict method must be implemented in subclass")

        def fit_predict(self, X, y=None):
            self.fit(X, y)
            return self.predict(X)
        
    class IPCA(ModelTemplate):

        def __init__(self, n_components=None):
            self.n_components = n_components
            self.model = IncrementalPCA(n_components=self.n_components,
                                        batch_size=self.n_components)

        def fit(self, X, y=None):
            self.model.fit(X, y)
            return self

        def predict(self, X):
            X = self.model.transform(X)
            return self.model.inverse_transform(X)
        
    class PCA(ModelTemplate):
        
        def __init__(self, n_components=None):
            self.n_components = n_components
            self.model = PCA(n_components=self.n_components)

        def fit(self, X, y=None):
            self.model.fit(X, y)
            return self

        def predict(self, X):
            X = self.model.transform(X)
            return self.model.inverse_transform(X)





