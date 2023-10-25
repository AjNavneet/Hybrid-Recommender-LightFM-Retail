import pandas as pd 
import numpy as np  

# Function to create a list of unique users from a specified column in the data
def unique_users(data, column):
    return np.sort(data[column].unique())  # Return a sorted array of unique user IDs

# Function to create a list of unique products/items from a specified column in the data
def unique_items(data, column):
    item_list = data[column].unique()  # Get the unique items from the specified column
    return item_list

# Function to create a list of features by concatenating specified columns from the customer data
def features_to_add(customer, column1, column2, column3):
    customer1 = customer[column1]
    customer2 = customer[column2]
    customer3 = customer[column3]
    combined_features = pd.concat([customer1, customer3, customer2], ignore_index=True).unique()
    return combined_features  # Return a unique list of concatenated features

# Function to create ID mappings for users, items, and features
def mapping(users, items, features):
    user_to_index_mapping = {}  # Initialize an empty dictionary to map user IDs to indices
    index_to_user_mapping = {}  # Initialize an empty dictionary to map indices to user IDs
    for user_index, user_id in enumerate(users):
        user_to_index_mapping[user_id] = user_index
        index_to_user_mapping[user_index] = user_id
        
    item_to_index_mapping = {}  # Initialize an empty dictionary to map item IDs to indices
    index_to_item_mapping = {}  # Initialize an empty dictionary to map indices to item IDs
    for item_index, item_id in enumerate(items):
        item_to_index_mapping[item_id] = item_index
        index_to_item_mapping[item_index] = item_id
        
    feature_to_index_mapping = {}  # Initialize an empty dictionary to map feature IDs to indices
    index_to_feature_mapping = {}  # Initialize an empty dictionary to map indices to feature IDs
    for feature_index, feature_id in enumerate(features):
        feature_to_index_mapping[feature_id] = feature_index
        index_to_feature_mapping[feature_index] = feature_id
        
    return user_to_index_mapping, index_to_user_mapping, \
           item_to_index_mapping, index_to_item_mapping, \
           feature_to_index_mapping, index_to_feature_mapping
