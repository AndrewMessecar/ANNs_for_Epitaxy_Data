# Andrew S. Messecar, 2023

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split , GridSearchCV
import pandas as pd
from random import seed
from itertools import product
from numpy import arange , empty
from pylab import show , figure, xlabel, ylabel, show, title, xticks, yticks, xlim, ylim
import seaborn as sb
seed(42)
Maximum_Iterations = 2000
Data = pd.read_csv(r'# Path to MoS2_Monolayer.csv')
Training_Data = pd.DataFrame( Data, columns = [ 'Ramp Up Duration (sec)' , 'Ramp Up Rotational Flow (sccm)' , 'Ramp Up Mo(CO)6 Flow (sccm)' , 'Pre-Growth Anneal #1 Duration (sec)' , 
                                         'Pre-Growth Anneal #1 Mo(CO)6 Flow (sccm)' , 'Pre-Growth Anneal #2 Duration (sec)' , 'Pre-Growth Anneal #2 Mo(CO)6 Flow (sccm)' , 
                                         'Pre-Growth Annealing Chalcogen Injector H2 Flow (sccm)' , 'Pre-Growth Temp (°C)' , 'H2S Flow (sccm)' , 'Growth Duration (seconds)' , 
                                         'Growth Temp (°C)' , 'Growth Metal Injector H2 Flow (sccm)' , 'Growth Chalcogen Injector H2 Flow (sccm)', 'Growth & Annealing Rotational Flow (sccm)' , 
                                         'Mo(CO)6 Flow (sccm)' , 'Post-Growth Anneal Duration (sec)' , 'Post-Growth Anneal Temp (°C)' , 'Post-Growth Anneal Reactor Pressure (torr)' , 
                                         'Cooldown #1 Duration (sec)' , 'Cooldown Reactor Pressure (torr)' , 'Cooldown #1 Rotational Flow (sccm)' , 'Cooldown Mo(CO)6 Flow (sccm)' , 
                                         'Cooldown #2 Duration (sec)' , 'Cooldown #2 Rotational Flow (sccm)' , 'Peak Difference'] )
Training_Data = Training_Data.sample( frac = 1 )
Training_and_Testing_Data = Training_Data.iloc[ :500 , : ]
Inputs = Training_and_Testing_Data.loc[ : , [ 'Ramp Up Duration (sec)' , 'Ramp Up Rotational Flow (sccm)' , 'Ramp Up Mo(CO)6 Flow (sccm)' , 'Pre-Growth Anneal #1 Duration (sec)' , 
                                         'Pre-Growth Anneal #1 Mo(CO)6 Flow (sccm)' , 'Pre-Growth Anneal #2 Duration (sec)' , 'Pre-Growth Anneal #2 Mo(CO)6 Flow (sccm)' , 
                                         'Pre-Growth Annealing Chalcogen Injector H2 Flow (sccm)' , 'Pre-Growth Temp (°C)' , 'H2S Flow (sccm)' , 'Growth Duration (seconds)' , 
                                         'Growth Temp (°C)' , 'Growth Metal Injector H2 Flow (sccm)' , 'Growth Chalcogen Injector H2 Flow (sccm)', 'Growth & Annealing Rotational Flow (sccm)' , 
                                         'Mo(CO)6 Flow (sccm)' , 'Post-Growth Anneal Duration (sec)' , 'Post-Growth Anneal Temp (°C)' , 'Post-Growth Anneal Reactor Pressure (torr)' , 
                                         'Cooldown #1 Duration (sec)' , 'Cooldown Reactor Pressure (torr)' , 'Cooldown #1 Rotational Flow (sccm)' , 'Cooldown Mo(CO)6 Flow (sccm)' , 
                                         'Cooldown #2 Duration (sec)' , 'Cooldown #2 Rotational Flow (sccm)' ] ]
Output = Training_and_Testing_Data.loc[ : , [ 'Peak Difference' ] ]
Training_Inputs, Testing_Inputs, Training_Output, Testing_Output = train_test_split( Inputs, Output, test_size = 0.2 , random_state = 42 )
Validation_Data = Training_Data.iloc[ 500: , : ]
Validation_Inputs = Validation_Data.loc[ : , [ 'Ramp Up Duration (sec)' , 'Ramp Up Rotational Flow (sccm)' , 'Ramp Up Mo(CO)6 Flow (sccm)' , 'Pre-Growth Anneal #1 Duration (sec)' , 
                                         'Pre-Growth Anneal #1 Mo(CO)6 Flow (sccm)' , 'Pre-Growth Anneal #2 Duration (sec)' , 'Pre-Growth Anneal #2 Mo(CO)6 Flow (sccm)' , 
                                         'Pre-Growth Annealing Chalcogen Injector H2 Flow (sccm)' , 'Pre-Growth Temp (°C)' , 'H2S Flow (sccm)' , 'Growth Duration (seconds)' , 
                                         'Growth Temp (°C)' , 'Growth Metal Injector H2 Flow (sccm)' , 'Growth Chalcogen Injector H2 Flow (sccm)', 'Growth & Annealing Rotational Flow (sccm)' , 
                                         'Mo(CO)6 Flow (sccm)' , 'Post-Growth Anneal Duration (sec)' , 'Post-Growth Anneal Temp (°C)' , 'Post-Growth Anneal Reactor Pressure (torr)' , 
                                         'Cooldown #1 Duration (sec)' , 'Cooldown Reactor Pressure (torr)' , 'Cooldown #1 Rotational Flow (sccm)' , 'Cooldown Mo(CO)6 Flow (sccm)' , 
                                         'Cooldown #2 Duration (sec)' , 'Cooldown #2 Rotational Flow (sccm)' ] ]
Validation_Output = Validation_Data.loc[ : , [ 'Peak Difference' ] ]

Input_Scaler = MinMaxScaler()
Input_Scaler.fit( Training_Inputs )
Training_Inputs = Input_Scaler.transform( Training_Inputs )
Testing_Inputs = Input_Scaler.transform( Testing_Inputs )
Validation_Inputs = Input_Scaler.transform( Validation_Inputs )
First_Layer_Neurons = arange(1 , 10 , 1); Second_Layer_Neurons = arange(1 , 10 , 1); Third_Layer_Neurons = arange(1 , 10 , 1)
Hidden_Layer_Sizes = list( product( First_Layer_Neurons , Second_Layer_Neurons ) )
# Parameters = {'hidden_layer_sizes' : [ [ 1 , 9 ] , [ 1 , 9 ] , [ 1 , 9 ] ] , 'activation' : ('identity' , 'logistic' , 'tanh' , 'relu' ) , 'solver' : ( 'lbfgs' , 'sgd' , 'adam' ) , 'alpha' : [ 10 ** (-1) , 10 ** (-7) ] , 'batch_size' : [1 , 1520]}
Parameters = { 'hidden_layer_sizes' : Hidden_Layer_Sizes , 'activation' : ('identity' , 'logistic' , 'tanh' , 'relu' ) , 'solver' : ( 'lbfgs' , 'sgd' , 'adam' ) } 

Artificial_Neural_Network_Without_Early_Stopping = MLPRegressor( learning_rate = 'adaptive' , shuffle = 1 , verbose = 0 , early_stopping = 0 , max_iter = Maximum_Iterations , random_state = 42)

Optimized_ANN_Without = GridSearchCV( estimator = Artificial_Neural_Network_Without_Early_Stopping , param_grid = Parameters , n_jobs = -1)

Optimized_ANN_Without.fit( Training_Inputs , Training_Output)

ANN_With_Early_Stopping = MLPRegressor( learning_rate = 'adaptive' , shuffle = 1 , verbose = 0 , early_stopping = 1 , max_iter = Maximum_Iterations , random_state = 42)

Optimal_ANN_With = GridSearchCV( estimator = ANN_With_Early_Stopping , param_grid = Parameters , n_jobs = -1)

Optimal_ANN_With.fit( Training_Inputs , Training_Output)

print( "\n Results from Grid Search without Early Stopping " )
print( "\n The best estimator across ALL searched params:\n" , Optimized_ANN_Without.best_estimator_ , "\n")
print( "\n The best score across ALL searched params:\n" , Optimized_ANN_Without.best_score_ , "\n")
print( "\n The best parameters across ALL searched params:\n" , Optimized_ANN_Without.best_params_ , "\n")

print( "\n Results from Grid Search with Early Stopping " )
print( "\n The best estimator across ALL searched params:\n" , Optimal_ANN_With.best_estimator_ , "\n")
print( "\n The best score across ALL searched params:\n" , Optimal_ANN_With.best_score_ , "\n")
print( "\n The best parameters across ALL searched params:\n" , Optimal_ANN_With.best_params_ , "\n")

# Mapping_Data = pd.read_csv(r'# Path to Mapping Data')
# Map_Space = pd.DataFrame( Mapping_Data, columns = [ 'Ramp Up Duration (sec)' , 'Ramp Up Rotational Flow (sccm)' , 'Ramp Up Mo(CO)6 Flow (sccm)' , 'Pre-Growth Anneal #1 Duration (sec)' , 
#                                         'Pre-Growth Anneal #1 Mo(CO)6 Flow (sccm)' , 'Pre-Growth Anneal #2 Duration (sec)' , 'Pre-Growth Anneal #2 Mo(CO)6 Flow (sccm)' , 
#                                         'Pre-Growth Annealing Chalcogen Injector H2 Flow (sccm)' , 'Pre-Growth Temp (°C)' , 'H2S Flow (sccm)' , 'Growth Duration (seconds)' , 
#                                         'Growth Temp (°C)' , 'Growth Metal Injector H2 Flow (sccm)' , 'Growth Chalcogen Injector H2 Flow (sccm)', 'Growth & Annealing Rotational Flow (sccm)' , 
#                                         'Mo(CO)6 Flow (sccm)' , 'Post-Growth Anneal Duration (sec)' , 'Post-Growth Anneal Temp (°C)' , 'Post-Growth Anneal Reactor Pressure (torr)' , 
#                                         'Cooldown #1 Duration (sec)' , 'Cooldown Reactor Pressure (torr)' , 'Cooldown #1 Rotational Flow (sccm)' , 'Cooldown Mo(CO)6 Flow (sccm)' , 
#                                         'Cooldown #2 Duration (sec)' , 'Cooldown #2 Rotational Flow (sccm)' ] )

# Map_Space = Input_Scaler.transform(Map_Space)
# Map = empty([1000,1000])
# Map = Optimized_ANN.predict(Map_Space)

# figure(dpi=1200)
# sb.heatmap(Map.reshape(1000, 1000))
# xlim([899,1000])
# ylim(50)
# xlabel("Growth Mo(CO)6 Injector Hydrogen Flow (sccm)")
# ylabel("Ramp Up Mo(CO)6 Flow (sccm)")
# yticks(arange(-0.01, 50, step=5), ['50', '45', '40', '35', '30', '25', '20', '15', '10', '5', '0'])
# xticks(arange(899.0, 1000, step=10), ['900','910','920','930','940','950', '960', '970', '980', '990', '1000'],rotation=0)
# title("Probability of GaN Growing Single Crystalline")
# xlabel("Initial Nitrogen Pressure (microtorr)")
# ylabel("Substrate Temperature (Celsius)")

# show()
