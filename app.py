from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, precision_score, recall_score

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load data
df = pd.read_csv('restaurant_recommendations.csv')
df.fillna('', inplace=True)
df['Cuisine'] = df['Cuisine'].astype(str).str.lower().str.strip()
df['Location'] = df['Location'].astype(str).str.lower().str.strip()
df['combined_features'] = df['Cuisine'] + ' ' + df['Location'] + ' ' + df['Reviews'].astype(str)

# TF-IDF & Cosine Similarity
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# KNN Collaborative Filtering
knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
knn_model.fit(tfidf_matrix)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    cuisines = sorted(df['Cuisine'].unique())
    error = None
    metrics = {}

    if request.method == 'POST':
        restaurant_name = request.form.get('restaurant')
        min_rating = float(request.form.get('min_rating', 0))
        cuisine_filter = request.form.get('cuisine')

        matches = df[df['Restaurant Name'].str.lower() == restaurant_name.lower()]
        if matches.empty:
            error = "Restaurant not found"
        else:
            idx = matches.index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
            recommended_indices = [i[0] for i in sim_scores]

            recommendations = df.iloc[recommended_indices]

            # Apply filters
            if min_rating:
                recommendations = recommendations[recommendations['Rating'] >= min_rating]
            if cuisine_filter and cuisine_filter.lower() != "all":
                recommendations = recommendations[recommendations['Cuisine'] == cuisine_filter.lower()]

            # Metrics
            metrics = calculate_metrics(matches, recommendations)

    return render_template('index.html', recommendations=recommendations, cuisines=cuisines, error=error, metrics=metrics)

def calculate_metrics(actual_df, recommended_df):
    actual_ratings = actual_df['Rating'].values
    predicted_ratings = recommended_df['Rating'].values[:len(actual_ratings)]

    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    threshold = 4
    actual_relevant = (actual_ratings >= threshold).astype(int)
    predicted_relevant = (predicted_ratings >= threshold).astype(int)

    precision = precision_score(actual_relevant, predicted_relevant, average='binary', zero_division=0)
    recall = recall_score(actual_relevant, predicted_relevant, average='binary', zero_division=0)

    return {'RMSE': rmse, 'Precision': precision, 'Recall': recall}

@app.route('/charts')
def charts():
    # Basic charts using Seaborn
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Rating'], bins=20, kde=True)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('static/rating_distribution.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    top_cuisines = df['Cuisine'].value_counts().head(5)
    sns.barplot(x=top_cuisines.index, y=top_cuisines.values)
    plt.title('Top 5 Cuisines')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('static/top_cuisines.png')
    plt.close()


    return render_template("charts.html", images=['rating_distribution.png', 'top_cuisines.png'])

if __name__ == '__main__':
    app.run(debug=True)
