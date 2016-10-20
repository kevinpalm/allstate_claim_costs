from utilities import read_data, column_check
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest



def get_estimator():
    """Helper function so that the estimator parameters aren't stored in multiple places"""
    estimator = Pipeline([("normalize", Normalizer()),
                          ("regress", RandomForestRegressor())])
    return estimator


def local_test(train_x, train_y):
    """Do a local test with simple train/test splitting"""

    # Split up the data
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y)

    # Define estimator and train
    estimator = get_estimator()
    estimator.fit(train_x, train_y)

    # Predict and score
    predicts = estimator.predict(test_x)
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
        train_y = train["loss"].tolist()
        train_x = train.drop("loss", axis=1)

        # Get dummies and check for misfit dummies
        train_x = pd.get_dummies(train_x)
        test_x = pd.get_dummies(test)
        train_x, test_x = column_check(train_x, test_x)

        # Pickle data files
        train_x.to_pickle("../input/train_x.pkl")
        train_y.to_pickle("../input/train_y.pkl")
        test_x.to_pickle("../input/test_x.pkl")

    # # Do a local test
    # local_test(train_x, train_y)

if __name__ == '__main__':
    main()