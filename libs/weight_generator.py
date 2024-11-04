from libs.data_manipulation import *
from libs.feature_generation import *
from libs.dim_reduction import *
from libs.football_plots import *
from libs.clustering import *
from libs.convex_hull import *

import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.stats import wasserstein_distance_nd
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
        #THINK OF OTHER WAY TO DO THIS! TALK TO JON
        #weights = np.array(weights)
        #
        ## Shift weights so that the smallest weight is 0
        #min_weight = np.min(weights)
        #weights = weights - min_weight
        #
        ## Scale weights to the range [0, 1] by dividing by the new max
        #max_weight = np.max(weights)
        #if max_weight > 0:
        #    weights = weights / max_weight
        #
        ## Normalize weights to sum to 1
        #weights_sum = np.sum(weights)
        #if weights_sum > 0:
        #    weights = weights / weights_sum
        #
        ## Correct the last weight to ensure exact sum of 1
        #if weights.size > 0:
        #    weights[-1] += 1 - np.sum(weights)
        #
        #weights = np.abs(weights)
        #weights_list.append(weights.tolist())
    
    return weights_list  # Return a list of arrays with normalized weights


def normalize_positions_with_ball(df):
    # Create copies of ball x and y columns for both home and away teams
    df_normalized = df.copy()
    
    # Normalize 'home' team player positions
    home_columns_x = [col for col in df.columns if col.startswith('home') and col.endswith('_x')]
    home_columns_y = [col for col in df.columns if col.startswith('home') and col.endswith('_y')]
    
    for x_col, y_col in zip(home_columns_x, home_columns_y):
        df_normalized[x_col] = df[x_col] - df['ball_x']
        df_normalized[y_col] = df[y_col] - df['ball_y']
    
    # Normalize 'away' team player positions
    away_columns_x = [col for col in df.columns if col.startswith('away') and col.endswith('_x')]
    away_columns_y = [col for col in df.columns if col.startswith('away') and col.endswith('_y')]
    
    for x_col, y_col in zip(away_columns_x, away_columns_y):
        df_normalized[x_col] = df[x_col] - df['ball_x']
        df_normalized[y_col] = df[y_col] - df['ball_y']
    
    return df_normalized

def most_similar_with_wasserstein(relevant_index, relevant_df, weighting_function, steps = 48):
    print(relevant_index)
    one_match = relevant_df
    identified_corner_df= relevant_df.iloc[relevant_index:relevant_index+1]
    one_match = one_match.iloc[::steps]
    #####
    inverse_identified_corner_weights = calculate_weights(identified_corner_df, fun = weighting_function)
    inverse_distance_list = calculate_weights(one_match, fun= weighting_function) #Inverse proportionality to distance
    one_match = normalize_positions_with_ball(one_match)
    coordinates_numpy = one_match.filter(regex ="^home").to_numpy()
    identified_corner_coordinates_numpy = identified_corner_df.filter(regex ="^home").to_numpy()
    #####

    
    identified_corner_coordinates = [list(zip(row[~np.isnan(row)][::2],row[~np.isnan(row)][1::2])) for row in identified_corner_coordinates_numpy]
    coordinates_zipped = [list(zip(row[~np.isnan(row)][::2],row[~np.isnan(row)][1::2])) for row in coordinates_numpy]
    
    #Get closest situations
    distances = []

    i = 0
    for weights, coordinates in zip(inverse_distance_list, coordinates_zipped):
        i+=1

        if(not np.isnan(np.sum(weights)) and len(weights) == 11 and len( coordinates) == 11):
            distances.append((wasserstein_distance_nd(identified_corner_coordinates[0], coordinates, u_weights= inverse_identified_corner_weights[0], v_weights=weights), i))
    indices_and_distances = sorted(distances, key = lambda t: t[0])
    indices = [index for _,index in indices_and_distances]
    return indices