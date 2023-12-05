# Andrew S. Messecar, 2023

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split , GridSearchCV
import pandas as pd
from random import seed
from itertools import product
from numpy import arange
seed(42)
Maximum_Iterations = 2000
Data = pd.read_csv(r' # Path to InN_Crystal.csv')
Training_Data = pd.DataFrame( Data, columns = [ 'Substrate Temperature (Celsius)' , 'Indium Temperature (Celsius)' , 'Chamber Pressure (torr)' , 'RF Plasma Power (W)' , 'Growth Time (minutes)' , 'Crystal' ] )
Training_Data = Training_Data.sample( frac = 1 )
Training_and_Testing_Data = Training_Data.iloc[ :100 , : ]
Inputs = Training_and_Testing_Data.loc[ : , [ 'Substrate Temperature (Celsius)' , 'Indium Temperature (Celsius)' , 'Chamber Pressure (torr)' , 'RF Plasma Power (W)' , 'Growth Time (minutes)' ] ]
Output = Training_and_Testing_Data.loc[ : , [ 'Crystal' ] ]
Training_Inputs, Testing_Inputs, Training_Output, Testing_Output = train_test_split( Inputs, Output, test_size = 0.2 , random_state = 42 )
Validation_Data = Training_Data.iloc[ 100: , : ]
Validation_Inputs = Validation_Data.loc[ : , [ 'Substrate Temperature (Celsius)' , 'Indium Temperature (Celsius)' , 'Chamber Pressure (torr)' , 'RF Plasma Power (W)' , 'Growth Time (minutes)' ] ]
Validation_Output = Validation_Data.loc[ : , [ 'Crystal' ] ]
Input_Scaler = MinMaxScaler()
Input_Scaler.fit( Training_Inputs )
Training_Inputs = Input_Scaler.transform( Training_Inputs )
Testing_Inputs = Input_Scaler.transform( Testing_Inputs )
Validation_Inputs = Input_Scaler.transform( Validation_Inputs )
First_Layer_Neurons = arange(1 , 10 , 1); Second_Layer_Neurons = arange(1 , 10 , 1); # Third_Layer_Neurons = arange(1 , 10 , 1)
Hidden_Layer_Sizes = list( product( First_Layer_Neurons , Second_Layer_Neurons ) )
# Parameters = {'hidden_layer_sizes' : [ [ 1 , 9 ] , [ 1 , 9 ] , [ 1 , 9 ] ] , 'activation' : ('identity' , 'logistic' , 'tanh' , 'relu' ) , 'solver' : ( 'lbfgs' , 'sgd' , 'adam' ) , 'alpha' : [ 10 ** (-1) , 10 ** (-7) ] , 'batch_size' : [1 , 1520]}
Parameters = { 'hidden_layer_sizes' : Hidden_Layer_Sizes , 'activation' : ('identity' , 'logistic' , 'tanh' , 'relu' ) , 'solver' : ( 'lbfgs' , 'sgd' , 'adam' ) } 

Artificial_Neural_Network = MLPRegressor( learning_rate = 'adaptive' , shuffle = 1 , verbose = 0 , early_stopping = 0 , max_iter = Maximum_Iterations , random_state = 42)

Optimized_ANN = GridSearchCV( estimator = Artificial_Neural_Network , param_grid = Parameters , n_jobs = -1)

Optimized_ANN.fit( Training_Inputs , Training_Output)

print( "\n Results from Grid Search without Early Stopping " )
print( "\n The best estimator across ALL searched params:\n" , Optimized_ANN.best_estimator_ , "\n")
print( "\n The best score across ALL searched params:\n" , Optimized_ANN.best_score_ , "\n")
print( "\n The best parameters across ALL searched params:\n" , Optimized_ANN.best_params_ , "\n")

ANN = MLPRegressor( learning_rate = 'adaptive' , shuffle = 1 , verbose = 0 , early_stopping = 1 , max_iter = Maximum_Iterations , random_state = 42)

Optimal_ANN = GridSearchCV( estimator = ANN , param_grid = Parameters , n_jobs = -1)

Optimal_ANN.fit( Training_Inputs , Training_Output)

print( "\n Results from Grid Search with Early Stopping " )
print( "\n The best estimator across ALL searched params:\n" , Optimal_ANN.best_estimator_ , "\n")
print( "\n The best score across ALL searched params:\n" , Optimal_ANN.best_score_ , "\n")
print( "\n The best parameters across ALL searched params:\n" , Optimal_ANN.best_params_ , "\n")
