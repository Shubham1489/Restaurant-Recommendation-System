import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('restaurant_recommendations.csv')

# Preview the data
print(df.head())

# Fill missing values with empty strings
df['Cuisine'] = df['Cuisine'].fillna('')
df['Location'] = df['Location'].fillna('')
df['Reviews'] = df['Reviews'].fillna('')

# Combine features into a single string
df['combined_features'] = df['Cuisine'] + " " + df['Location'] + " " + df['Reviews'].astype(str)

# Convert text data into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_restaurant(restaurant_name, num_recommendations=5):
    if restaurant_name not in df['Restaurant Name'].values:
        return f"Restaurant '{restaurant_name}' not found in database."

    # Get index of the restaurant
    idx = df[df['Restaurant Name'] == restaurant_name].index[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    sim_scores = sim_scores[1:num_recommendations+1]
    recommended_indices = [i[0] for i in sim_scores]
    
    return df[['Restaurant Name', 'Rating', 'Cuisine', 'Location']].iloc[recommended_indices]

# Example usage
recommendations = recommend_restaurant("The Spice House")
print(recommendations)
