import numpy as np 

# Function to get recommendations for a user using a recommendation model
def get_recommendations(model, user, items, user_to_product_interaction_matrix, user2index_map, product_to_feature_interaction_matrix):
    # Get the user's index
    user_index = user2index_map.get(user, None)
    
    # If the user doesn't exist in the mapping, return None
    if user_index is None:
        return None
    
    # Retrieve the user's index
    users = user_index
    
    # Get products already bought by the user
    known_positives = items[user_to_product_interaction_matrix.tocsr()[user_index].indices]
    print('User index =', users)
    
    # Predict scores using the model
    scores = model.predict(user_ids=users, item_ids=np.arange(user_to_product_interaction_matrix.shape[1]), item_features=product_to_feature_interaction_matrix)
    
    # Get top recommended items based on scores
    top_items = items[np.argsort(-scores)]
    
    # Print out the results
    print("User %s" % user)
    print("     Known positives:")  # Already bought items
    for x in known_positives[:10]:
        print("                  %s" % x)
    
    print("     Recommended:")  # Items recommended to the user
    for x in top_items[:10]:
        print("                  %s" % x)
