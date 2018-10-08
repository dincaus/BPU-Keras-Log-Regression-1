import pandas as pd
import numpy as np

from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.regularizers import L1L2

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

NUMBER_OF_FEATURES = 21
EPOCHS = 600
BATCH_SIZE = 32

value_scale = StandardScaler()

# load dataset
dataset = pd.read_csv('../data/FIFA_2018_Statistics.csv')

# let's clean and prepare dataset
dataset = dataset.drop(['Own goal Time', '1st Goal', 'Team', 'Date', 'Round', 'Opponent'], axis=1)

dataset['PSO'] = pd.get_dummies(dataset.PSO).Yes
dataset['Man of the Match'] = pd.get_dummies(dataset['Man of the Match']).Yes

# add new column which will shows who is the 'winner'
dataset['winner'] = dataset.groupby(
    np.repeat(
        [
            n for n in range(len(dataset) // 2)
        ], 2
    )
)['Goal Scored'].transform(lambda x: x == max(x))

dataset['winner'] = dataset['winner'].map({True: 1, False: 0})

# get train and test data
X = value_scale.fit_transform(dataset.drop('Man of the Match', axis=1))
X[np.isnan(X)] = 0
Y = dataset['Man of the Match']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.30)
Y_train = np_utils.to_categorical(Y_train, 2)
Y_test = np_utils.to_categorical(Y_test, 2)

if __name__ == "__main__":

    optimizer = RMSprop(0.001)
    model = Sequential()

    model.add(
        Dense(
            2,
            activation="softmax",
            input_dim=NUMBER_OF_FEATURES,
            kernel_regularizer=L1L2(l1=0.1, l2=0.01)
        )
    )

    model.compile(
        optimizer,
        "categorical_crossentropy",
        ['accuracy']
    )

    model.summary()

    model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_data=(X_test, Y_test),
        verbose=2
    )
