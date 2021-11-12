import numpy as np


def generate_example_timeseries(num_examples, length):
    """Generates a dataset of three-channel timeseries of desired size,
    meant to substitute (but not exactly approximate) wearable sensor data.
    First two channels represent heart rate and stepcount, modelled
    as cyclical trends with the same periodicity, offset from each other.
    Third channel represents data missingness, where no sensor data
    was recorded."""

    timeseries = []
    for i in range(num_examples):
        # we require two noisy timeseries channels
        # that are related in their periodicity

        # we'll approximate this with sinusoids.

        # random starting phase:
        phase = np.random.uniform(0, 2*np.pi)
        # and periodicity:
        period = np.random.uniform(1, 5)

        cycle1 = np.sin(period * np.linspace(phase, 10+phase, length))
        # second cycle is delayed from the first by a random amount:
        delay = np.random.normal(loc=1.0, scale=0.2)
        cycle2 = np.sin(period * np.linspace(phase+delay, 10+phase+delay, length))

        # heart rate as a noisy sinusoid:
        heartrate = np.random.normal(scale=0.3, size=length) + cycle1
        # normalise heartrate to unit variance:
        heartrate /= np.std(heartrate)

        # step count as a noisy sinusoid, sat back from heart rate, with floor of 0:
        steps = np.maximum(np.random.normal(scale=0.2, size=length) + cycle2, 0)
        # rescale steps to set maximum equal to 1:
        steps /= np.max(steps)

        # third channel:
        # define a random block of missing data for each sample
        # by picking some indices of where missing data starts and ends
        nonmissingness = np.ones(length)

        missing_data_start = int(np.random.uniform(0, length//2))
        missing_data_end = int(np.random.uniform(missing_data_start, missing_data_start + length//2))
        nonmissingness[missing_data_start:missing_data_end] = 0

        # zero out other channels when data is missing:
        heartrate[missing_data_start:missing_data_end] = 0
        steps[missing_data_start:missing_data_end] = 0


        timeseries.append(np.stack([heartrate, steps, nonmissingness], axis=1))

    # compose dense three-channel time series data:
    timeseries_arr = np.stack(timeseries, axis=0)

    return timeseries_arr
