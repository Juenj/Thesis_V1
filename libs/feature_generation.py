import numpy as np
import pandas as pd




def calculate_msd(df, window_size):
    """
    Calculate the mean squared displacement for a set of points.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the points.
    window_size (int): The window size for which to calculate the mean squared displacement.

    Returns:
    np.array: The mean squared displacement values for the points.
    """
    
    df = df.filter(regex='^home')
    # Number of time steps
    np_data = df.to_numpy()
    n = len(df.index)
    # Initialize the MSD array
    if (window_size > n):
        return np.array([])
    # Loop over all time lags
    msds = []
    #Goes through each column
    for j in range(0, np_data.shape[1] - 1, 2):
        msd = np.zeros(window_size)
        #Changes initial start point
        for i in range(n-window_size):
            #Goes through each lag
            for lag in range(1, window_size):
                if pd.notna(np_data[i, j]) and pd.notna(np_data[i+lag,j]) and pd.notna(np_data[i, j+1]) and pd.notna(np_data[i+lag,j+1]):

                        # Calculate the squared displacement for the current time lag
                    displacement = (np_data[i, j] - np_data[i+lag,j])**2 + \
                                       (np_data[i, j+1] - np_data[i+lag,j+1])**2 

                    msd[lag] += np.mean(displacement)
        msd = msd/window_size 
        msds.append(msd)

    return np.array(msds)


def msd_for_dataframe(df, indices, max_time_lag):
    """
    Calculate the mean squared displacement for a set of points.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the points.
    indices (list): The indices of the rows in the DataFrame.
    max_time_lag (int): The maximum time lag for which to calculate the mean squared displacement.

    Returns:
    np.array: The mean squared displacement values for the points.
    """

    msds = []
    for index in indices:
        print(index)
        working_data = df.loc[index:index + max_time_lag].iloc[::12]
        if (len(working_data[working_data["half"] == working_data.iloc[0]["half"]]) == len(working_data.index)):
            msd_data = calculate_msd(working_data)
       
            # Number of particles and time lags
            msds.append(msd_data)

 
    msds_stacked = np.vstack(msds)
    
    return msds_stacked





import numpy as np
import pandas as pd

def ripley_k(points: pd.Series, radii: np.linspace, width: float, height: float):
    """
    Ripley's K function for a set of points in a 2D area.

    Parameters:
    points (pd.Series): The points for which to calculate Ripley's K values,
                        expected format: [x_1, y_1, x_2, y_2, ...].
    radii (np.linspace): The radii for which to calculate Ripley's K values.
    width (float): The width of the area in which the points are located.
    height (float): The height of the area in which the points are located.

    Returns:
    list: The Ripley's K values for the points. 
    """
    
    # Reshape points from flat Series to array of (x, y) pairs
    n = len(points) // 2  # Since points come in pairs (x, y)
    points_array = np.array(points).reshape(n, 2)

    area = width * height
    lambda_density = n / area
    k_values = []

    # Loop through each radius value
    for r in radii:
        count = 0
        
        # Loop through each point and calculate the pairwise distances
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate Euclidean distance between point i and point j
                    distance = np.linalg.norm(points_array[i] - points_array[j])
                    if distance < r:
                        count += 1

        # Calculate Ripley's K for the given radius
        k_r = count / (n * lambda_density)
        k_values.append(k_r)

    return k_values



def ripley_k_by_indices(df, indices):
    """
    Calculate Ripley's K values for a set of points.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the points.
    indices (list): The indices of the rows in the DataFrame.

    Returns:
    np.array: The Ripley's K values for the points.
    """
    k_vals = np.array([ripley_k(df.filter(regex='^home').loc[i],np.arange(0, 34), 105.0, 68.0) for i in indices])
    return k_vals

#HOME_1_x, 