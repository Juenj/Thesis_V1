import os
import pandas as pd
import numpy as np

def standardize_columns(df, team_label, home=True):
    """
    Standardize columns to 'home' or 'away' depending on the team's role in the match.
    
    Parameters:
    - df: DataFrame containing tracking data.
    - team_label: 'home' or 'away', indicating the team role.
    - home: Boolean, if True, treat as home team, else away team.
    
    Returns:
    - The DataFrame with standardized column names.
    """
    if home:
        new_columns = {col: col.replace(f'{team_label}_', 'home_') for col in df.columns if f'{team_label}_' in col}
    else:
        new_columns = {col: col.replace(f'{team_label}_', 'away_') for col in df.columns if f'{team_label}_' in col}
    
    df.rename(columns=new_columns, inplace=True)
    return df

def compile_team_tracking_data(base_directory, team_name):
    """
    Compile all tracking data for a given team across all matches into one large CSV.
    
    Parameters:
    - base_directory: The directory containing all match folders.
    - team_name: The name of the team for which to compile data.
    
    Returns:
    - A DataFrame with all the tracking data combined, and also saves it as a CSV file.
    """
    compiled_df = pd.DataFrame()
    folders = os.listdir(base_directory)
    # sort folders
    folders = sorted(folders)

    for folder_name in folders:
        folder_path = os.path.join(base_directory, folder_name)
        
        # Ensure it's a directory
        if os.path.isdir(folder_path):
            # Determine if the team is playing at home or away
            if team_name in folder_name:
                if folder_name.endswith(team_name) or folder_name.endswith(team_name.upper()):
                    # Team is away
                    home_team = False
                else:
                    # Team is home
                    home_team = True
                
                # Load the correct CSVs
                if home_team:
                    team_csv = os.path.join(folder_path, 'tracking_home.csv')
                    opp_csv = os.path.join(folder_path, 'tracking_away.csv')
                else:
                    team_csv = os.path.join(folder_path, 'tracking_away.csv')
                    opp_csv = os.path.join(folder_path, 'tracking_home.csv')

                # Read the CSVs
                if os.path.exists(team_csv) and os.path.exists(opp_csv):
                    team_df = pd.read_csv(team_csv)
                    opp_df = pd.read_csv(opp_csv)

                    # Standardize column names for both dataframes
                    team_df = standardize_columns(team_df, 'home' if home_team else 'away', home=True)
                    opp_df = standardize_columns(opp_df, 'home' if not home_team else 'away', home=False)

                    # Ensure unique column names before concatenation
                    team_df.columns = [f"{col}_team" if col in opp_df.columns else col for col in team_df.columns]
                    opp_df.columns = [f"{col}_opp" if col in team_df.columns else col for col in opp_df.columns]

                    # Combine the DataFrames column-wise
                    combined_df = pd.concat([team_df, opp_df], axis=1)



             
                    # Append to the compiled DataFrame
                    compiled_df = pd.concat([compiled_df, combined_df], ignore_index=True, axis=0)

    # Save the compiled data to a CSV
    output_csv_path = os.path.join(base_directory, f"{team_name}_compiled_tracking_data.csv")
    compiled_df.to_csv(output_csv_path, index=False)

    return compiled_df


def rename_columns(df, team):
    """ Rename columns in the DataFrame to standardize player number columns. 
    
    Parameters:
    - df: DataFrame containing tracking data.
    - team: Team label, either 'home' or 'away'.

    Returns:
    - The DataFrame with standardized column names.
    """
    # Rename columns to match standard player number columns
    new_columns = {col: col.replace(f'{team}_', 'player_') for col in df.columns}
    df.rename(columns=new_columns, inplace=True)

def compile_den_csvs(base_directory):
    """
    Compile all tracking data for Denmark across all matches into one large DataFrame.

    Parameters:
    base_directory (str): The directory containing all match folders.

    Returns:
    pd.DataFrame: The compiled DataFrame.
    """

    # Initialize an empty DataFrame to hold all data
    compiled_df = pd.DataFrame()

    # Loop through all directories in the base directory
    for folder_name in os.listdir(base_directory):
        # Check if the folder name contains "Denmark" or "DEN"
        if 'Denmark' in folder_name or 'DEN' in folder_name:
            folder_path = os.path.join(base_directory, folder_name)
            
            # Ensure it is a directory
            if os.path.isdir(folder_path):
                # Split the folder name by underscores
                parts = folder_name.split('_')
                
                # Determine whether Denmark/DEN is the home or away team
                if parts[-1] in ['Denmark', 'DEN']:
                    # If Denmark/DEN is the last part, use tracking_away.csv
                    csv_path = os.path.join(folder_path, 'tracking_away.csv')
                    team = 'away'
                else:
                    # If Denmark/DEN is not the last part, use tracking_home.csv
                    csv_path = os.path.join(folder_path, 'tracking_home.csv')
                    team = 'home'
                
                # Check if the selected file exists
                if os.path.exists(csv_path):
                    # Read the CSV
                    temp_df = pd.read_csv(csv_path)
                    # Rename columns to standardize player number columns
                    rename_columns(temp_df, team)
                    # Append the DataFrame to the compiled DataFrame
                    compiled_df = pd.concat([compiled_df, temp_df], ignore_index=True)

    return compiled_df



def calculate_possession(df: pd.DataFrame):
    """
    Calculate possession based on the closest player to the ball.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    pd.DataFrame: The DataFrame with the additional 'Possession' column.
    """

    # Initialize a list to store possession results
    possession_list = []
    
    # Identify columns that match home and away player coordinates
    home_columns = [col for col in df.columns if col.startswith('home_') and col.endswith('_x')]
    away_columns = [col for col in df.columns if col.startswith('away_') and col.endswith('_x')]
    
    # Iterate through each row (tick)
    for index, row in df.iterrows():
        # Get ball position
        ball_x, ball_y = row['ball_x'], row['ball_y']
        
        # Calculate distances to home players
        home_distances = []
        for home_col in home_columns:
            jersey_number = home_col.split('_')[1]  # Extract jersey number
            home_x = row[f'home_{jersey_number}_x']
            home_y = row[f'home_{jersey_number}_y']
            if pd.notna(home_x) and pd.notna(home_y):
                distance = np.sqrt((home_x - ball_x)**2 + (home_y - ball_y)**2)
                home_distances.append(distance)
        
        # Calculate distances to away players
        away_distances = []
        for away_col in away_columns:
            jersey_number = away_col.split('_')[1]  # Extract jersey number
            away_x = row[f'away_{jersey_number}_x']
            away_y = row[f'away_{jersey_number}_y']
            if pd.notna(away_x) and pd.notna(away_y):
                distance = np.sqrt((away_x - ball_x)**2 + (away_y - ball_y)**2)
                away_distances.append(distance)
        
        # Find the closest home and away player
        closest_home_distance = min(home_distances) if home_distances else float('inf')
        closest_away_distance = min(away_distances) if away_distances else float('inf')
        
        # Determine possession based on the closest player
        if closest_home_distance < closest_away_distance:
            possession = 'Home'
        elif closest_away_distance < float('inf'):
            possession = 'Away'
        else:
            possession = 'None'
        
        # Append the result to the list
        possession_list.append(possession)
    
    # Add the possession list to the DataFrame at once
    df['Possession'] = possession_list
    
    return df

def generate_ball_xy_delta(df: pd.DataFrame):
    """
    Generate the absolute delta x and y coordinates for the ball.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    pd.DataFrame: The DataFrame with the additional columns.
    """
    ball_delta_x = np.diff(df["ball_x"].to_numpy(), prepend = [0])

    ball_delta_y = np.diff(df["ball_y"].to_numpy(), prepend = [0])
    df["ball_delta_x"] = np.abs(ball_delta_x)
    df["ball_delta_y"] = np.abs(ball_delta_y)
    return df
    

def extract_and_save_coordinates(df : pd.DataFrame, index_value : int, output_filename : str):
    """
    Extracts the x and y coordinates for a specific index from the DataFrame, aligns them,
    and saves the result to a CSV file.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    index_value (int): The index for which the coordinates need to be extracted.
    output_filename (str): The name of the output CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the aligned x and y coordinates.
    """

    
    # Filter columns for x and y coordinates
    x_columns = df.filter(regex='_x$').columns
    y_columns = df.filter(regex='_y$').columns

    # Extract and align x and y coordinates
    coords = pd.DataFrame({
        'x': [df.loc[index_value, col] for col in x_columns],
        'y': [df.loc[index_value, col.replace('_x', '_y')] for col in y_columns]
    })

    # Drop any rows with NaN values
    coords = coords.dropna()

    # Save to CSV
    coords.to_csv(output_filename, index=False)

    return coords



def append_features(data):
    """
    Append additional features to the DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.

    Returns:
    pd.DataFrame: The DataFrame with additional features appended.
    """
    
    data = calculate_possession(data)
    data = generate_ball_xy_delta(data)
    return data



def extract_one_match(df: pd.DataFrame, num_matches=1, tick_distance =1):
    """
    Extracts data for the specified number of matches from the DataFrame.
    A new match is identified by a reset of Time [s] to zero.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the match data.
    num_matches (int): The number of matches to extract. Defaults to 1.

    Returns:
    pd.DataFrame: A DataFrame containing data for the specified number of matches.
    """
    
    # Identify indices where Time [s] resets to zero
    match_start_indices = df[df["Time [s]"] == 0].index.tolist()
    match_start_indices.append(len(df))  # Append the last index for the end of the final match

    # Extract data for the specified number of matches
    if num_matches > len(match_start_indices) - 1:
        print(f"Warning: Only {len(match_start_indices) - 1} matches are available. Returning all matches.")
        num_matches = len(match_start_indices) - 1
    
    match_data = df.iloc[match_start_indices[0]:match_start_indices[num_matches]]

    # Select every 24th tick but ensure there are no missing ticks in the range
    if len(match_data) % tick_distance != 0:
        print(f"Warning: Missing some ticks, only selecting up to the nearest multiple of 24.")
        match_data = match_data.iloc[:-(len(match_data) % tick_distance)]  # Drop the remaining ticks not divisible by 24
    
    # Select every 24th tick
    match_data = match_data.iloc[::tick_distance]
    # Reset the index
    match_data.reset_index(drop=True, inplace=True)
    # drop the columns that are nan but skip the first row   
    match_data = match_data.dropna(axis=1, how='all', subset=match_data.index[1:])
    
    return match_data