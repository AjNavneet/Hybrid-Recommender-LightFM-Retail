# Import required libraries
import pandas as pd  
import numpy as np   
from lightfm import LightFM  # LightFM for building hybrid recommendation models
from lightfm.evaluation import auc_score  
import time  

# Function to build a hybrid model with different loss functions
def hybrid_model(loss, interaction_train, product_interaction):
    if loss == 'warp':  # Loss function = WARP (Weighted Approximate-Rank Pairwise)
        model_with_features = LightFM(loss="warp")  # Initialize the LightFM model with WARP loss
        start = time.time()  # Record the start time

        model_with_features.fit_partial(interaction_train, 
            user_features=None, 
            item_features=product_interaction, 
            sample_weight=None, 
            epochs=1, 
            num_threads=4,
            verbose=False)  # Fit the model with partial data

        end = time.time()  # Record the end time
        print("time taken for fitting = {0:.{1}f} seconds".format(end - start, 2))
        return model_with_features

    elif loss == 'logistic':  # Loss function = logistic
        model_with_features = LightFM(loss="logistic", no_components=30)  # Initialize the LightFM model with logistic loss and 30 components
        start = time.time()

        model_with_features.fit_partial(interaction_train,
            user_features=None, 
            item_features=product_interaction, 
            sample_weight=None, 
            epochs=10, 
            num_threads=20,
            verbose=False)  # Fit the model with partial data

        end = time.time()
        print("time taken for fitting = {0:.{1}f} seconds".format(end - start, 2))
        return model_with_features

    elif loss == 'bpr':  # Loss function = BPR (Bayesian Personalized Ranking)
        model_with_features = LightFM(loss="bpr")  # Initialize the LightFM model with BPR loss
        start = time.time()

        model_with_features.fit_partial(interaction_train,
            user_features=None, 
            item_features=product_interaction, 
            sample_weight=None, 
            epochs=1, 
            num_threads=4,
            verbose=False)  # Fit the model with partial data

        end = time.time()
        print("time taken = {0:.{1}f} seconds".format(end - start, 2))
        return model_with_features

    else:
        print("Invalid loss function specified")

# Function to evaluate the model with AUC score
def evaluate_model(model, interaction_test, interaction_train, product_interaction):
    start = time.time()  # Record the start time

    auc_with_features = auc_score(model=model,  
        test_interactions=interaction_test,
        train_interactions=interaction_train, 
        item_features=product_interaction,
        num_threads=4,
        check_intersections=False)  # Calculate AUC score

    end = time.time()  # Record the end time

    print("time taken for AUC score = {0:.{1}f} seconds".format(end - start, 2))

    return "average AUC without adding item-feature interaction = {0:.{1}f}".format(auc_with_features.mean(), 2)
