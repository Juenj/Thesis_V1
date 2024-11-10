from fastdtw import fastdtw
from scipy.spatial.distance import euclidean 
def find_similar_movement(df, index_of_target_movement, indices_of_candidate_sequeces, length_of_sequence=120, columns=["ball_x", "ball_y"]):
    """
    Finds and ranks sequences in a DataFrame based on similarity to a target sequence.
    
    Parameters:
    - df: DataFrame containing the data of interest, such as coordinates for a ball.
    - index_of_target_movement: The starting index in the DataFrame for the target movement sequence.
    - indices_of_candidate_sequeces: List of starting indices in the DataFrame for candidate sequences to compare.
    - length_of_sequence: The number of frames to consider for each sequence. Default is 120.
    - columns: List of column names to extract coordinates (default is ["ball_x", "ball_y"]).
    
    Returns:
    - A list of tuples containing the distance and index of each candidate sequence, ranked by similarity to the target.
    """
    
    distances = []  # Initialize list to store distances between the target sequence and each candidate.
    
    # Extract and normalize the target sequence.
    target_sequence = df[columns].loc[index_of_target_movement:index_of_target_movement+length_of_sequence:12].to_numpy()
    target_sequence = target_sequence - target_sequence[0]  # Normalize by setting the first point as the origin.

    # Loop through each candidate sequence to compare with the target sequence.
    for index in indices_of_candidate_sequeces:
        print(index)  # Print index for tracking the process
        
        # Extract and normalize the current candidate sequence.
        considered_sequence = df[columns].loc[index:index+length_of_sequence:12].to_numpy()
        considered_sequence = considered_sequence - considered_sequence[0]  # Normalize by setting the first point as the origin.

        # Compute the distance between the target and candidate sequence using fast dynamic time warping (fastdtw).
        try:
            distance, path = fastdtw(target_sequence, considered_sequence, dist=euclidean)
        except:
            # Skip this candidate if an error occurs (e.g., due to sequence length mismatch).
            continue

        # Append the distance and the candidate index to the distances list.
        distances.append((distance, index))

    return distances
