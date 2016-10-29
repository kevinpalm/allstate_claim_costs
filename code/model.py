from utilities import read_data, column_check
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import tensorflow as tf


def nnet_predict(trX, trY, teX):

    # Fix label shape
    trY = np.reshape(np.array(trY), [len(trY), 1])

    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def model(X, w_h, w_h2, w_o, p_keep_input,
              p_keep_hidden):
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu(tf.matmul(X, w_h))

        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h2))

        h2 = tf.nn.dropout(h2, p_keep_hidden)

        return tf.matmul(h2, w_o)

    X = tf.placeholder("float", [None, 1079])
    Y = tf.placeholder("float", [None, 1])

    w_h = init_weights([1079, 860])
    w_h2 = init_weights([860, 430])
    w_o = init_weights([430, 1])

    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()

        count = 0
        for i in range(10):
            count += 1
            for start, end in zip(range(0, len(trX), 800), range(800, len(trX) + 1, 800)):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                              p_keep_input: 0.8, p_keep_hidden: 0.5})
            print("Session {} complete".format(count))

        predicts = sess.run(py_x, feed_dict={X: teX, p_keep_input: 1.0, p_keep_hidden: 1.0})
        return predicts

def get_estimator():
    """Helper function so that the estimator parameters aren't stored in multiple places"""
    estimator = RandomForestRegressor()
    return estimator


def local_test(train_x, train_y):
    """Do a local test with simple train/test splitting"""

    # Split up the data
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y)

    # # Define estimator and train
    # estimator = get_estimator()
    # estimator.fit(train_x, train_y)
    #
    # # Predict and score
    # predicts = estimator.predict(test_x)
    predicts = nnet_predict(train_x, train_y, test_x)
    print("Mean Absolute Error for the local test was {}.".format(mean_absolute_error(test_y, predicts)))


def main():

    try:
        # Read in the pickle files if they exist already
        train_x = pd.read_pickle("../input/train_x.pkl")
        train_y = pd.read_pickle("../input/train_y.pkl")
        test_x = pd.read_pickle("../input/test_x.pkl")

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

    # Do a local test
    local_test(train_x, train_y)

if __name__ == '__main__':
    main()