import numpy as np
import random

def scramble_timeseries(timeseries):
    """Expects a dense NxTx3 array of three-channel timeseries data,
    with N samples of duration T.

    If the three channels are: heartrate, stepcount, nonmissingness,
    returns two new datasets.

    The first returned value is the 'scrambled' dataset where heartrate
    and stepcount are permuted between the N samples, so that the heartrate
    channel of each sample now corresponds to the stepcount channel of another.

    The second returned value is an adjusted form of the original dataset,
    with the exception that it now has the same amount of missing data as in
    the scrambled dataset.

    This is necessary because the scrambling process has to incorporate the
    missingness values of both the heartrate and stepcount channels, meaning
    it would have more missing data on average than the original data without
    this adjustment.

    The adjusted original dataset is returned in the same order as the original
    dataset. The scrambled dataset's heartrate channel is in the same order
    as the original, but the stepcount channel is permuted randomly."""

    real_array = np.asarray(timeseries)

    real_hr = real_array[:,:,0]
    real_steps = real_array[:,:,1]
    real_nonmissing = real_array[:,:,2]

    # preallocate data for shuffled step counts before shuffling:
    shuf_steps = real_steps.copy()
    # keep response patterns associated with the unshuffled HR and the shuffled steps:
    # real_hr_resp = np.stack([real_hr, real_nonmissing], axis=2)
    shuf_steps_resp = np.stack([shuf_steps, real_nonmissing], axis=2)

    # we need a special shuffle such that no element ends up in the same place it
    # was before (since then the 'scrambled' sample is not actually scrambled):

    np.random.shuffle(shuf_steps_resp)
    shuf_steps = shuf_steps_resp[:,:,0]

    # combine response pattern channels into their sum:
    combined_nonmissing = real_nonmissing * shuf_steps_resp[:,:,1] # becomes 0 (missing) if either is 0

    fake_array = np.stack([real_hr, shuf_steps, combined_nonmissing], axis=2)

    # now scrambled data has significantly more missing periods than the real data
    # so the model may learn to distinguish real from fake simply by counting missingness periods

    # so, to ensure the amount of missiness is the same in both, we interpose the missingness
    # of the scrambled data back onto the real data (which is therefore, in a sense, no longer 'real')

    real_array[:,:,2] = combined_nonmissing

    return fake_array, real_array

def format_training_set(fake_array, real_array):
    """Expects scrambled and adjusted-original array of dense timeseries data.
    Simply combines them into a single shuffled array, and returns that
    alongside a second array of labels denoting scrambled versus original."""

    # format data as x,y pair for network training:
    x = np.concatenate([real_array, fake_array])
    y = np.asarray([0.]*len(real_array) + [1.]*len(fake_array)).reshape(-1, 1) # represents fakeness
    total_samples = len(x)
    assert len(y) == total_samples

    # shuffle dataset:
    rand_idxs = random.sample(range(total_samples), total_samples)

    x = x[rand_idxs] # data/examples
    y = y[rand_idxs] # labels/targets

    return x, y
