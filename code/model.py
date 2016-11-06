from utilities import read_data, column_check
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def pca_step(train_x, test_x):
    """Apply pca to the feature datasets"""

    # Define pca
    decomp = Pipeline([("rscale", RobustScaler()),
                       ("mmscale", MinMaxScaler()),
                       ("pca", IncrementalPCA(n_components=540))])

    # Fit and transform
    decomp.fit(train_x)

    return pd.DataFrame(decomp.transform(train_x), index=train_x.index),\
           pd.DataFrame(decomp.transform(test_x), index=test_x.index)


def nnet_predict(trX, trY, teX, teY=None):
    """Get predictions using a dropout deep neural network"""

    # Fix label shape
    trY = np.reshape(np.array(trY), [len(trY), 1])

    def init_weights(shape):  # helper function for initializing weights
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden, bias):  # defining the network

        # Dropout on input layer, flow to hidden layer 1
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu6(tf.matmul(X, w_h))

        # Dropout on hidden layer 1, flow to hidden layer 2
        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu6(tf.matmul(h, w_h2))

        # Dropout on hidden layer 2, return output
        h2 = tf.nn.dropout(h2, p_keep_hidden)

        return tf.add(tf.matmul(h2, w_o), bias)

    # Initialize placeholders for input features and output values
    X = tf.placeholder("float", [None, 540])
    Y = tf.placeholder("float", [None, 1])

    # Initialize variables for weights and bias
    w_h = init_weights([540, 432])
    w_h2 = init_weights([432, 432])
    w_o = init_weights([432, 1])
    bias = tf.Variable(tf.to_float(trY.mean()))

    # Initialize placeholders for dropout parameters
    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    # Name model
    py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden, bias=bias)

    # Define cost as mean absolute error
    cost = tf.reduce_mean(tf.abs(tf.sub(py_x, Y)))

    # Prep for learning rate decay
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               270, 0.7, staircase=True)

    # Train to minimize mean absolute error
    # Passing global_step to minimize() will increment it at each step.
    train_op = (tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step))

    # Launch the graph in a session
    with tf.Session() as sess:

        # you need to initialize all variables
        tf.initialize_all_variables().run()

        # Count so we can report how many trainings we've finished
        count = 0

        # Set how many trainings we'll do
        for i in range(10):
            count += 1
            shuffle(trX, trY)

            # Divide the data into batches
            batchsize = int(len(trX)/340)
            for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX) + 1, batchsize)):

                # Train on the batch
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                              p_keep_input: 0.8, p_keep_hidden: 0.5})

            print("Training {} complete".format(count))

            # If true outcomes were supplied (as in a local cross validation test), go ahead and report loss too
            if teY is not None:
                predicts = sess.run(py_x, feed_dict={X: teX, p_keep_input: 1.0, p_keep_hidden: 1.0})
                print("Mean Absolute Error for round was {}.".format(mean_absolute_error(teY, predicts)))

        # Get final predictions
        predicts = sess.run(py_x, feed_dict={X: teX, p_keep_input: 1.0, p_keep_hidden: 1.0})

        return predicts


def local_test(train_x, train_y):
    """Do a local test with simple train/test splitting"""

    # Split up the data
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y)

    predicts = nnet_predict(train_x, train_y, test_x, test_y)
    print("Mean Absolute Error for the local test was {}.".format(mean_absolute_error(test_y, predicts)))


def main():

    try:
        # Read in the pickle files if they exist already
        train_x = pd.read_pickle("../input/train_x_pc.pkl")
        train_y = pd.read_pickle("../input/train_y.pkl")
        test_x = pd.read_pickle("../input/test_x_pc.pkl")

    except:
        # Read in the data
        train, test = read_data()

        # Pull out the outcomes
        train_y = train["loss"]
        train_x = train.drop("loss", axis=1)

        # Get dummies and check for misfit dummies
        train_x = pd.get_dummies(train_x)
        test_x = pd.get_dummies(test)
        train_x, test_x = column_check(train_x, test_x)

        # Pickle data files
        train_x.to_pickle("../input/train_x.pkl")
        train_y.to_pickle("../input/train_y.pkl")
        test_x.to_pickle("../input/test_x.pkl")

        # Decomposition
        train_x, test_x = pca_step(train_x, test_x)

        # Pickle principle component data files
        train_x.to_pickle("../input/train_x_pc.pkl")
        test_x.to_pickle("../input/test_x_pc.pkl")

    # Do a local test
    print("Starting local test...")
    local_test(train_x, train_y)

    print("Starting final model...")
    test_x["loss"] = nnet_predict(train_x, train_y, test_x)
    test_x[["loss"]].to_csv("../output/decomp_nn_predict.csv")



if __name__ == '__main__':
    main()