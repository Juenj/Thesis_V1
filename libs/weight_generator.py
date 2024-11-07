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


def calculate_weights(df: pd.DataFrame, normalizing_factor = 11, ball_x_col='ball_x', ball_y_col='ball_y', regex="^home", fun=lambda x: x, max_val =1):
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
            
           
            # Calculate the distance to the ball and ensure a small epsilon is added
            distance_to_ball = np.sqrt((player_x - ball_x[frame_idx])**2 + (player_y - ball_y[frame_idx])**2)
          
            weight = fun(distance_to_ball)  # Add epsilon to ensure positivity
            if (weight > 190):
                print(weight)
            weights.append(np.min([weight, max_val])/normalizing_factor)
        weights.append(1-np.sum(weights)) #Adding final weight for ball
        weights_list.append(weights)

  
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

def most_similar_with_wasserstein(relevant_index, relevant_df, weighting_function, steps = 48, normalizing_factor = 11, max_weight = 1):
    one_match = relevant_df
    identified_corner_df= relevant_df.iloc[relevant_index:relevant_index+1]
    one_match = one_match.iloc[::steps]
    #####
    inverse_identified_corner_weights = calculate_weights(identified_corner_df,normalizing_factor, fun = weighting_function, max_val=max_weight)
    inverse_distance_list = calculate_weights(one_match,normalizing_factor, fun= weighting_function, max_val=max_weight) #Inverse proportionality to distance
    #one_match = normalize_positions_with_ball(one_match)

    # Filter the columns, then reorder so 'ball_x_team' and 'ball_y_team' are last
    columns_to_select = one_match.filter(regex="^home|ball_x_team|ball_y_team").columns
    # Separate ball_x_team and ball_y_team columns and place them at the end
    reordered_columns = [col for col in columns_to_select if not col.startswith("ball")] + \
                        [col for col in columns_to_select if col.startswith("ball")]
    

    # Apply the reordered columns to the DataFrame, then convert to numpy
    coordinates_numpy = one_match[reordered_columns].to_numpy()
    
    print(one_match[reordered_columns].head())


    # Repeat the same process for identified_corner_df
    columns_to_select_identified = identified_corner_df.filter(regex="^home|ball_x_team|ball_y_team").columns
    reordered_columns_identified = [col for col in columns_to_select_identified if not col.startswith("ball")] + \
                                   [col for col in columns_to_select_identified if col.startswith("ball")]
    
    print(identified_corner_df[reordered_columns_identified])
    print(inverse_identified_corner_weights[0])
    identified_corner_coordinates_numpy = identified_corner_df[reordered_columns_identified].to_numpy()

    #####

    
    identified_corner_coordinates = [list(zip(row[~np.isnan(row)][::2],row[~np.isnan(row)][1::2])) for row in identified_corner_coordinates_numpy]
    coordinates_zipped = [list(zip(row[~np.isnan(row)][::2],row[~np.isnan(row)][1::2])) for row in coordinates_numpy]
    
    #Get closest situations
    distances = []
    indices = one_match.index.to_numpy()
    i = 0
    for weights, coordinates in zip(inverse_distance_list, coordinates_zipped):
        
        if(not np.isnan(np.sum(weights)) and (len(weights) == len(inverse_identified_corner_weights[0])) and (len(coordinates) == len(identified_corner_coordinates[0]) )):
            
            
            distances.append((wasserstein_distance_nd(identified_corner_coordinates[0], coordinates, u_weights= inverse_identified_corner_weights[0], v_weights=weights), indices[i]))
        i+=1
    indices_and_distances = sorted(distances, key = lambda t: t[0])
    indices = [index for _,index in indices_and_distances]
    return indices




def filter_by_ball_radius(data, index, radius):
    # Get the ball position at the specified index
    ref_ball_x = data.at[index, 'ball_x_team']
    ref_ball_y = data.at[index, 'ball_y_team']
    
    # Calculate the distance of each row's ball position from the reference position
    distances = np.sqrt((data['ball_x_team'] - ref_ball_x)**2 + (data['ball_y_team'] - ref_ball_y)**2)
    
    # Filter rows where the distance is less than or equal to the radius
    filtered_data = data[distances <= radius]
    
    return filtered_data