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
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class Regressor():

    def __init__(self, x, nb_epoch = 1000):
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
        x.fillna(value=values)

        # Use one hot encoding to encode the ocean proximity column
        transformer = make_column_transformer((OneHotEncoder(), ['ocean_proximity']), remainder='passthrough')
        transformed = transformer.fit_transform(x)
        transformed_x = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
        
        # Normalise the dataset
        normalised_x = (transformed_x - transformed_x.min()) / (transformed_x.max() - transformed_x.min())

        if y is not None: 
            self.y_min = y.min()[0]
            self.y_max = y.max()[0]
            normalised_y = np.array((y - self.y_min) / (self.y_max - self.y_min))
        else: normalised_y = None

        return np.array(normalised_x), normalised_y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
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

        # X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        # return self

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget # This will give us the training dataset for x and y.
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=2, shuffle=True)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32) # Might have to reshape these, check later.
        y_val = torch.tensor(y_val, dtype=torch.float32) # Might have to reshape these, check later.

        # How are we defining the model...

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 18),
            nn.ReLU(),
            nn.Linear(18, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        ) # Arbitrary numbers...

        loss_func = nn.MSELoss()
        
        # Choose Adam or AdaDelta as an optimiser.
        # Experiment..?
        # yes

        optimiser = torch.optim.Adam(self.model.parameters(), lr=0.001) # Interesting...

        # Method 2
        # optimiser = torch.optim.Adam() # Start with default values for the optimiser, chose adam because recommended in lectures...

        train_dataset = TensorDataset(X_train, y_train)
        self.batch_size = 20
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size) # ASSUMING DATA IS ALREADY SHUFFLED!!

        for epoch in range(self.nb_epoch):
            self.model.train()

            for batch_inputs, batch_target in train_loader:
                optimiser.zero_grad() # Turns gradients to zero?
                outputs = self.model(batch_inputs)
                loss = loss_func(outputs, batch_target)
                loss.backward()
                optimiser.step()

            # outputs = self.model(x)
            avg_training_loss = loss.item() / len(train_loader)
            print(f"Epoch {epoch+1}, Average training loss: {avg_training_loss}")

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
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # X, _ = self._preprocessor(x, training = False) # Do not forget
        # pass

        X, _ = self._preprocessor(x, training = False) # Do not forget
        
        self.eval()

        with torch.no_grad():
            xTensor = torch.tensor(X.values, dtype=torch.float32)
            yHat = self(xTensor) 
            return yHat.numpy() if torch.is_tensor(yHat) else yHat.detach().numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget

        mse = mean_squared_error(X[:,-1], Y)
        rmse = np.sqrt(mse)

        print("MSE: ", mse, "\nRMSE: ", rmse)

        # # This plot SHOULD give a more intuitive visualisation of predicted vs true values. I have some doubts as to whther this is correct or not
        # plot_data = pd.DataFrame({'True Values': X[:, -1], 'Predicted Values': Y.flatten()})

        # plt.scatter(x=plot_data['Predicted Values'], y=plot_data['True Values'], color='red', label='True Values', s=10)
        # plt.scatter(x=plot_data['Predicted Values'], y=plot_data['Predicted Values'], color='blue', label='Predicted Values', s=10)

        # # sns.regplot(x='Predicted Values', y='True Values', data=plot_data, scatter=False, line_kws={'color': 'blue', 'label': 'Predicted Values'})

        # plt.legend()
        # plt.title('Regression Plot: True vs Predicted Values')
        # plt.show()

        return rmse 

        # options for regression tasks: mse, rmse
        # return 0

        # essentially display the loss. The method depends on what type of regression/output activation we're using. Wait is linear the only type that is regression 
        
        # This measures the average of the squares of errors or deviations, 
        # providing a relative measure of model performance in terms of how close predictions are to the actual values


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



def RegressorHyperParameterSearch(x_train, y_train, x_valid, y_valid, model): 
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
    params = {'batch_size':[8, 16, 32, 64],
              'nb_epoch':[200, 400, 600, 800, 1000]}
    gs = GridSearchCV(estimator=model, param_grid=params, cv=10)

    best_params = gs.fit(x_train, y_train)
    print("Hyperparam tuning best accuracy: " + str(gs.best_score_))

    # surely plotting a graph is better than running over a bunch of values. How? idk
    # the idea is to plot a graph of accuracy vs nb_epochs, with two lines representing training and validation
    # where validation accuracy starts dropping, thats optimal no. of epochs. aka early stopping
    # this solves one hyperparam. problem, maybe. This method could apply to other hparams too?
    # am I tired? yes. thanks for asking

    return best_params # Return the chosen hyper parameters

    # Note to self: Use held-out not cross-validation. So train/validation/test
    # We try a bunch of different hyperparameters on training
    # Then, we are picking the hyperparameters that have the best accuracy according to the validation split

    # Do we need confidence intervals? Check for P-hacking? Ehhh

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



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
