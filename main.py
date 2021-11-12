import tensorflow as tf
from sklearn.model_selection import train_test_split
from synthetic_data import generate_example_timeseries
from visualisation import plot_dense_sensordata, plot_learning_curve
from data_scrambling import scramble_timeseries, format_training_set

# this flag allows two options:
# if TRAIN == True, fit the unsupervised CNN on training data
#    then use it to generate embeddings for the test data.
# if TRAIN == False, load a pre-saved CNN instead
#    and generate embeddings on test data without training.
TRAIN = True

# generate some example data:
TIME_SERIES_LENGTH = 360
NUM_TRAINING_EXAMPLES = 10000
sensor_df = generate_example_timeseries(NUM_TRAINING_EXAMPLES, TIME_SERIES_LENGTH)
# not a realistic model but will do for demonstrative purposes.

# sanity check: visualise the three data channels
# of the first few samples:
plot_dense_sensordata(sensor_df[:5])

fake_data, real_data = scramble_timeseries(sensor_df)

# format as training data for model:
x, y = format_training_set(fake_data, real_data)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# define unsupervised CNN model:
if TRAIN:
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2)
    metrics = [tf.keras.metrics.BinaryAccuracy()]

    # note that best choices of hyperparameters and architecture
    # will depend on your dataset.
    # in particular, L2 regularisation or dropout may be useful.

    #### HYPERPARAMETERS:
    activation = tf.nn.relu
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    num_epochs = 50
    dil_rate = 2
    kernel_size = 21
    validation_fraction = 0.5

    #### ARCHITECTURE:
    net = tf.keras.Sequential()

    net.add(tf.keras.layers.Conv1D(8, kernel_size, activation=activation, dilation_rate=dil_rate, input_shape = x_train.shape[1:]))
    net.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2))

    net.add(tf.keras.layers.Conv1D(20, kernel_size, activation=activation, dilation_rate=dil_rate))

    net.add(tf.keras.layers.Conv1D(32, kernel_size, activation=activation, dilation_rate=dil_rate))
    net.add(tf.keras.layers.GlobalAveragePooling1D()) # this is the latent feature space
    # net.add(tf.keras.layers.Dropout(0.5))
    net.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    net.compile(optimizer, loss, metrics=metrics)
    print(net.summary())

    # train:
    hist = net.fit(x_train, y_train,
        batch_size=128,
        epochs=num_epochs,
        validation_split=validation_fraction,
        #validation_data=(x_test, y_test),
        shuffle=True,
        )

    # check how training went:
    plot_learning_curve(hist)

    # save model for later loading:
    net.save('cnn_rate_unsupervised_trained')

else: # load pre-saved model:
    net = tf.keras.models.load_model('cnn_rate_unsupervised_trained')

#### EMBEDDING:

# remove head from CNN so it terminates after global pooling,
# but before dropout:
embed_net = tf.keras.Sequential(net.layers[:-2])

# calculate embeddings from test data:
x_test_embeddings = embed_net(x_test).numpy()

# these can then be used in regression or other predictive models.
