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


def calculate_weights(df: pd.DataFrame, ball_x_col='ball_x', ball_y_col='ball_y', regex="^home", fun=lambda x: x, epsilon=1e-6):
    ball_x = df[ball_x_col].values
    ball_y = df[ball_y_col].values
    player_cols = df.filter(regex=regex).columns
    
    # Extract player positions and calculate adjusted inverse distance to the ball
    weights_list = []
    x_cols = [col for col in player_cols if col.endswith('_x')]
    y_cols = [col for col in player_cols if col.endswith('_y')]
    indices = df.index.to_numpy()

    for frame_idx in range(len(df)):
        weights = []
        for i in range(len(x_cols)):  # Loop through all players
            player_x = df.loc[indices[frame_idx], x_cols[i]]
            player_y = df.loc[indices[frame_idx], y_cols[i]]
            
            # Check if player_x or player_y is NaN (inactive player)
            if np.isnan(player_x) or np.isnan(player_y):
                continue  # Skip this player if they are inactive
            
            #SOMETHING WRONG HERE
            # Calculate the distance to the ball and ensure a small epsilon is added
            distance_to_ball = np.sqrt((player_x - ball_x[frame_idx])**2 + (player_y - ball_y[frame_idx])**2)
          
            weight = fun(distance_to_ball)  # Add epsilon to ensure positivity
            weights.append(weight)
        # Convert weights to a numpy array
 
        weights = np.array(weights)
        
        # Shift weights so that the smallest weight is 0
        min_weight = np.min(weights)
        weights = weights - min_weight
        
        # Scale weights to the range [0, 1] by dividing by the new max
        max_weight = np.max(weights)
        if max_weight > 0:
            weights = weights / max_weight
        
        # Normalize weights to sum to 1
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights = weights / weights_sum
        
        # Correct the last weight to ensure exact sum of 1
        if weights.size > 0:
            weights[-1] += 1 - np.sum(weights)
        
        weights = np.abs(weights)
        weights_list.append(weights.tolist())
    
    return weights_list  # Return a list of arrays with normalized weights
