from libs.data_manipulation import *
from libs.feature_generation import *
from libs.dim_reduction import *
from libs.football_plots import *
from libs.clustering import *
from libs.convex_hull import *

import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

from mplsoccer import *
import pandas as pd
import numpy as np
import os



# Function to calculate weights based on proximity to the ball and ensure they sum to exactly 1
def calculate_weights(df: pd.DataFrame, ball_x_col='ball_x', ball_y_col='ball_y', regex="^home", fun = lambda x : x):
    ball_x = df[ball_x_col].values
    ball_y = df[ball_y_col].values
    player_cols = df.filter(regex=regex).columns
    
    # Extract player positions and calculate inverse distance to the ball
    weights_list = []
    x_cols = [col for col in player_cols if col.endswith('_x')]
    y_cols = [col for col in player_cols if col.endswith('_y')]
    indices = df.index.to_numpy()
    print(x_cols)
    print(y_cols)
    for frame_idx in range(len(df)):
        weights = []
        for i in range(len(x_cols)):  # Loop through all players
            player_x = df.loc[indices[frame_idx], x_cols[i]]
            player_y = df.loc[indices[frame_idx], y_cols[i]]
            
            # Check if player_x or player_y is NaN (inactive player)
            if np.isnan(player_x) or np.isnan(player_y):
                continue  # Skip this player if they are inactive
            
            # Calculate the distance to the ball
            distance_to_ball = np.sqrt((player_x - ball_x[frame_idx])**2 + (player_y - ball_y[frame_idx])**2)
            weight = fun(distance_to_ball) #Custom function
            weights.append(weight)
        
        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights = weights / weights_sum  # Normalize
        
        # Correct the last weight to ensure exact sum of 1
        if weights.size > 0:
            weights[-1] += 1 - np.sum(weights)
        
        weights_list.append(weights.tolist())
    
    return weights_list  # Return a list of arrays with normalized weights


