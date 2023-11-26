import sys
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.utils import shuffle
# import seaborn as sns

class Regressor():

    def __init__(self, x, batch_size=10, learning_rate = 0.001, optimiser = "Adam", nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate 
        self.optimiser = optimiser
        # self.rho = rho
        # self.eps = eps
        # self.weight_decay = weight_decay
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        # return x, (y if isinstance(y, pd.DataFrame) else None)

        # Fill in the NaNs in the dataset with the column mean
        values = x[["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]].mean()
        values['ocean_proximity'] = 'INLAND'
        x = x.fillna(value=values)
        # Use one hot encoding to encode the ocean proximity column
        """transformer = make_column_transformer((OneHotEncoder(), ['ocean_proximity']), remainder='passthrough')
        transformed = transformer.fit_transform(x)
        transformed_x = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())"""
        x['ocean_proximity'] = pd.Categorical(x['ocean_proximity'], categories=['INLAND', '<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN', 'ISLAND'])
        transformed = pd.get_dummies(data=x, columns=['ocean_proximity'])
        transformed_x = np.vstack(transformed.values).astype(float)
        # print(np.isnan(transformed_x).any())
        tensor_x = torch.tensor(transformed_x, dtype=torch.float32)
        # Tensors??
        # tensor_x = torch.tensor(transformed_x.values, dtype=torch.float32)
        # print(torch.isnan(tensor_x).any())
        col_maximum = tensor_x.max(dim=0).values
        for i in range(len(col_maximum)):
            if col_maximum[i] == 0:
                col_maximum[i] = 1

        # print(temp)

        tensor_x = (tensor_x - tensor_x.min(dim=0).values) / (col_maximum - tensor_x.min(dim=0).values)
        # print(tensor_x.shape)       
        # print(torch.isnan(tensor_x).any())

        if y is not None:
            tensor_y = torch.tensor(y.values, dtype=torch.float32)
            self.y_min = tensor_y.min().item()
            self.y_max = tensor_y.max().item()
            tensor_y = (tensor_y - self.y_min) / (self.y_max - self.y_min)
        else:
            tensor_y = None

        return tensor_x, tensor_y
        
        # # Normalise the dataset
        # normalised_x = (transformed_x - transformed_x.min()) / (transformed_x.max() - transformed_x.min())

        # if y is not None: 
        #     self.y_min = y.min()[0]
        #     self.y_max = y.max()[0]
        #     normalised_y = np.array((y - self.y_min) / (self.y_max - self.y_min))
        # else: normalised_y = None

        # return np.array(normalised_x), normalised_y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def postprocess(self, y):
        denormalised_y = (y * (self.y_max - self.y_min)) + self.y_min
        return denormalised_y
        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X_train, y_train = self._preprocessor(x, y = y, training = True) # Do not forget # This will give us the training dataset for x and y.

        # How are we defining the model...
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 18),
            nn.ReLU(),
            nn.Linear(18, 14),
            nn.ReLU(),
            nn.Linear(14, 7),
            nn.ReLU(),
            nn.Linear(7, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        ) # Arbitrary numbers...

        loss_func = nn.MSELoss()

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay = 1e-5) # Interesting...
        scheduler = StepLR(optimiser, step_size=5, gamma=0.2) # To be tuned

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size) # ASSUMING DATA IS ALREADY SHUFFLED!!

        torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.nb_epoch):
            self.model.train()
            total_training_loss = 0

            for batch_inputs, batch_target in train_loader:
                optimiser.zero_grad() # Turns gradients to zero?
                outputs = self.model(batch_inputs)
                loss = loss_func(outputs, batch_target)
                loss.backward()
                optimiser.step()

                total_training_loss += loss.item()
            scheduler.step()

            # outputs = self.model(x)
        avg_training_loss = loss.item() / len(train_loader)
        # print(f"Epoch {epoch+1}, Average training loss: {avg_training_loss}")

        return self.model

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size], 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget

        with torch.no_grad():
            #self.model.eval()
            yHat = self.model(X)
            return self.postprocess(yHat.numpy() if torch.is_tensor(yHat) else yHat.detach().numpy())

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size). Run through predict to get predicted values
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1). True values

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # _, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        # trueValues = self.postprocess(Y.numpy())
        trueValues = y
        predictedValues = self.predict(x)

        mse = mean_squared_error(trueValues, predictedValues) # Use trueValues or y, depending on if I preprocess
        rmse = np.sqrt(mse)

        # print("MSE: ", mse, "\nRMSE: ", rmse)

        # # This plot SHOULD give a more intuitive visualisation of predicted vs true values.
        # Use trueValues or y, depending on if I preprocess

        # plt.scatter(x=predictedValues, y=trueValues, color='red', label='True Values', s=10) 
        # plt.plot(predictedValues, predictedValues, color='blue', label='Predicted Values', linestyle='-')
        # plt.xlabel("Predicted Values")
        # plt.ylabel("Values")
        # plt.legend()
        # plt.title('Regression Plot: True vs Predicted Values')
        # plt.show()

        return rmse 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(data): 

    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    output_label = "median_house_value"

    params = {'batch_size':[16, 24, 32, 40, 48],
              'nb_epoch':[5, 10, 15],
              'learning_rate':[0.0012],
              'opt':['Adam']}
    """gs = GridSearchCV(estimator=self.model, param_grid=params, cv=10)

    best_params = gs.fit(x, y)
    print("Hyperparam tuning best accuracy: " + str(gs.best_score_))"""

    best_params = {'nb_epoch':None, 'batch_size':None, 'learning_rate':None, 'opt':None}
    cur_best_score = None

    print("Epoch | Batch Size | Learning Rate | Optimiser | RMSE")
    for epoch in params['nb_epoch']:
        for size in params['batch_size']:
            for rate in params['learning_rate']:
                for opt in params['opt']:
                    data = data.sample(frac=1).reset_index(drop=True)

                    # Splitting input and output, and the dataset
                    total_rows = data.shape[0]
                    train_rows = round(0.7*total_rows)

                    x_train = data.loc[:train_rows, data.columns != output_label]
                    y_train = data.loc[:train_rows, [output_label]]

                    x_valid = data.loc[train_rows:, data.columns != output_label]
                    y_valid = data.loc[train_rows:, [output_label]]


                    regressor = Regressor(x_train, batch_size=size, learning_rate=rate, optimiser=opt, nb_epoch=epoch)

                    regressor.fit(x_train, y_train)
                    rmse = regressor.score(x_valid, y_valid)
                    print("[" + str(epoch) + "," + str(size) + "," + str(rate) + "," + opt + "," + str(rmse) + "],")
                    
                    if cur_best_score == None or rmse < cur_best_score:
                        best_params['nb_epoch'] = epoch
                        best_params['batch_size'] = size
                        best_params['learning_rate'] = rate
                        best_params['opt'] = opt
                        cur_best_score = rmse
                        save_regressor(regressor)


    return  best_params

    # Hyperparameters to tune: 
    # learning rate(recommended adaptive ones include Adam and AdaDelta, from the lecture), 
    # learning rate decay (from a cursory search, at least for Adam is not implemented, and is done seperately)
    # number of epochs, x
    # batch size, x
    # number of layers/neurons(need to combat overfitting via limiting capacity, early stopping, L1/L2 reguarisation, dropout), 
    # activation fns, 
    # weight initialisation (Xavier Groot probably)


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def train_main():

    output_label = "median_house_value"
    data = pd.read_csv("housing.csv") 
    data = data.sample(frac=1).reset_index(drop=True)

    # Splitting input and output, and the dataset
    total_rows = data.shape[0]
    train_rows = round(0.8*total_rows)

    x_train = data.loc[:train_rows, data.columns != output_label]
    y_train = data.loc[:train_rows, [output_label]]
    x_test = data.loc[train_rows:, data.columns != output_label]
    y_test = data.loc[train_rows:, [output_label]]

    # Fitting the model
    regressor = Regressor(x_train, batch_size=32, learning_rate=0.0012, optimiser='Adam', nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


def load_main():

    output_label = "median_house_value"
    data = pd.read_csv("housing.csv") 
    data = data.sample(frac=1).reset_index(drop=True)

    x_valid = data.loc[:, data.columns != output_label]
    y_valid = data.loc[:, [output_label]]

    # Training
    # make sure the model isn't overfitting
    regressor = load_regressor()

    # Error
    error = regressor.score(x_valid, y_valid)
    print("\nRegressor error: {}\n".format(error))

def hyperparam_main():
    data = pd.read_csv("housing.csv") 

    # Training and validation (for Hyperparam Tuning)
    print(RegressorHyperParameterSearch(data))


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) > 1:
            if sys.argv[1] == 'l':
                print("Loading the model pickle file")
                load_main()
            elif sys.argv[1] == 't':
                print("Running the neural network training method")
                train_main()
            elif sys.argv == 'h':
                print("Running the hyperparameter tuning method")
                hyperparam_main()
            else:
                print("Invalid command line parameter")
    else:
        print("Loading the model from a pickle file")
        load_main()
