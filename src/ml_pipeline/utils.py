import numpy as np 
import pandas as pd 
from scipy.sparse import coo_matrix  # Import coo_matrix from scipy.sparse for constructing sparse matrices

# Function to read data from an Excel file
def read_data(filepath, data):
    try:
        df = pd.read_excel(filepath, data)  # Read the Excel file into a DataFrame
    except Exception as e:
        print(e)  # Handle any exceptions that may occur during file reading
    else:
        return df  # Return the DataFrame if reading is successful

# Function to merge different data frames
def merge_dataset(df1, df2, left_on_param, right_on_param, join_type):
    try:
        final_df = pd.merge(df1, df2, left_on=left_on_param, right_on=right_on_param, how=join_type)
    except Exception as e:
        print(e)  # Handle any exceptions that may occur during data merging
    else:
        return final_df  # Return the merged data frame

# Function to create an interaction matrix from data
def interactions(data, row, col, value, row_map, col_map):
    # Map values from data to row and column indices using the provided mappings
    row = data[row].apply(lambda x: row_map[x]).values
    col = data[col].apply(lambda x: col_map[x]).values
    value = data[value].values
    
    # Create a coo_matrix (sparse matrix) using the mapped values
    return coo_matrix((value, (row, col)), shape=(len(row_map), len(col_map)))
