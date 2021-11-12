import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_dense_sensordata(dense_data, ax=None, show=True, show_missing=False, time=None, text=None, ylabels=True):
    """view single patient's worse of dense sensordata, a ndarray or df
    where the columns are: heart_rate, activity_rank, resp_pattern.
    Optionally supply a series of times of the same length as the array, for xlabel.
    If dense_data is a list or batched array, interpret it as a series of dense data arrays
        and plot all of them."""

    if isinstance(dense_data, list) or (len(dense_data.shape) == 3 and dense_data.shape[0] > 1):

        if len(dense_data) > 10:
            # display up to a max of 10 rows
            dense_data = dense_data[:10]

        num_plots = len(dense_data)
        fig = plt.figure()
        for p in range(num_plots):
            ax = fig.add_subplot(num_plots, 1, p+1)

            if isinstance(text, list): # interpret text as an input list of text strings
                this_text = text[p]
            else:
                this_text = None

            plot_dense_sensordata(dense_data[p], ax=ax, show=False, show_missing=show_missing, text=this_text, ylabels=ylabels)

        if show:
            plt.show()
        return

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()

    if isinstance(dense_data, pd.DataFrame):
        # cast to numpy values for simplicity
        dense_data = np.copy(dense_data.values)
    else:
        dense_data = np.copy(dense_data)

    if time is None:
        # default x axis is just a range
        time = np.arange(dense_data.shape[0])

    ax2 = ax.twinx() # secondary axis for heart rate
    # ax3 = ax.twinx() # tertiary axis for resp pattern

    resp_pattern = dense_data[:,2]
    missing = np.where(resp_pattern == 0)[0]

    # time = dense_data.index
    # heartrate:
    hr = dense_data[:,0]
    activity = dense_data[:,1]

    if not show_missing:
        activity[missing] = np.nan
        hr[missing] = np.nan

    ax.plot(time, hr, c='tab:orange')

    # activity (rank):
    ax2.plot(time, activity, c='tab:blue')

    # remove margin to ensure the background colour lines up:
    ax.margins(x=0)
    # ax2.margins(x=0)
    plt.xlim([0, len(dense_data)])

    # resp pattern:
    if len(missing) > 0:
        ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                  dense_data[:,2][np.newaxis],
                  cmap=plt.get_cmap('Reds').reversed(), alpha=0.3)

    # show y-labels
    if ylabels:
        ax.set_ylabel(f'Heartrate (orange)')
        ax2.set_ylabel(f'Activity (blue)')

    if text is not None:
        plt.text(0.05, 0.9, text,
            horizontalalignment='left',
            verticalalignment='center',
            transform = ax.transAxes)

    if show:
        plt.show()
    else:
        return ax


def plot_learning_curve(hist, epoch_start=0):
    plt.plot(hist.history['loss'][epoch_start:], c='red', linestyle='-')
    plt.plot(hist.history['val_loss'][epoch_start:], c='red', linestyle='--')
    plt.plot(hist.history['binary_accuracy'][epoch_start:], c='blue', linestyle='-')
    plt.plot(hist.history['val_binary_accuracy'][epoch_start:], c='blue', linestyle='--')
    plt.legend(['loss', 'val_loss', 'acc', 'val_acc'])
    plt.xlabel('Epoch'); plt.title('Learning curve for self-supervision task')
    # plt.ylim([0,0.2])
    plt.show()
