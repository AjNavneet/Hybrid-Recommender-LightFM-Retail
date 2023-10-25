import numpy as np  
from scipy.sparse import coo_matrix 

# Function to merge training and testing data into a single sparse matrix
def train_test_merge(training_data, testing_data):
    # Initialize a dictionary to store training data
    train_dict = {}
    
    # Populate the dictionary with training data (row, col) as keys and data as values
    for row, col, data in zip(training_data.row, training_data.col, training_data.data):
        train_dict[(row, col)] = data
    
    # Replace training data with testing data if it's greater (max of data values)
    for row, col, data in zip(testing_data.row, testing_data.col, testing_data.data):
        train_dict[(row, col)] = max(data, train_dict.get((row, col), 0))
    
    # Initialize lists to store row indices, column indices, and data values
    row_list = []
    col_list = []
    data_list = []
    
    # Populate the lists with data from the dictionary
    for row, col in train_dict:
        row_list.append(row)
        col_list.append(col)
        data_list.append(train_dict[(row, col)])
    
    # Convert lists to numpy arrays
    row_list = np.array(row_list)
    col_list = np.array(col_list)
    data_list = np.array(data_list)
    
    # Create a coo_matrix (sparse matrix) with the merged data
    return coo_matrix((data_list, (row_list, col_list)), shape=(training_data.shape[0], training_data.shape[1]))
