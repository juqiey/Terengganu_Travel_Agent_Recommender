from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Load the dataset
def load_data():
    dataset_path = '../Project/dataset_travel_agent.csv'
    df = pd.read_csv(dataset_path)
    df['rating'] = df['rating'].replace('no review yet', '0').astype(float)
    df['category'] = df['category'].fillna('')
    df['area'] = df['area'].fillna('')
    df['type'] = df['type'].fillna('')  # Assuming 'type' column exists
    df['validity'] = df['validity'].fillna('')  # Assuming 'validity' column exists
    df['content'] = df['name'] + ' ' + df['category'] + ' ' + df['area'] + ' ' + df['type'] + ' ' + df['validity'] + ' Rating:' + df['rating'].astype(str)
    return df

# Build the recommender system
def build_recommender(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['content'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Recommend travel agents
def recommend_travel_agents(input_query, df, cosine_sim, top_n=10):
    try:
        input_rating = float(input_query)
        filtered_df = df[df['rating'] >= input_rating]
        return filtered_df[['name', 'category', 'area', 'type', 'validity', 'rating']].head(top_n)
    except ValueError:
        if input_query in ['Kuala Terengganu', 'Kuala Nerus', 'Setiu', 'Besut', 'Marang', 'Kemaman', 'Dungun']:
            filtered_df = df[df['area'] == input_query]
            return filtered_df[['name', 'category', 'area', 'type', 'validity', 'rating']].head(top_n)
        else:
            vectorizer = TfidfVectorizer()
            input_tfidf = vectorizer.fit_transform([input_query])
            all_tfidf = vectorizer.transform(df['content'])

            cosine_sim_query = linear_kernel(input_tfidf, all_tfidf).flatten()
            sim_scores = list(enumerate(cosine_sim_query))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_indices = [i[0] for i in sim_scores[:top_n]]
            return df.iloc[sim_indices][['name', 'category', 'area', 'type', 'validity', 'rating']]

# Evaluate the recommender system
def evaluate_recommender(df, cosine_sim):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    test_indices = pd.Series(test.index, index=test['name']).drop_duplicates()
    mse_list = []

    for agent_name in test['name']:
        if agent_name in test_indices:
            idx = test_indices[agent_name]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_indices = [i[0] for i in sim_scores[1:6]]

            true_ratings = df.iloc[sim_indices]['rating']
            predicted_rating = true_ratings.mean()
            mse_list.append((df.loc[idx, 'rating'] - predicted_rating) ** 2)

    return np.sqrt(np.mean(mse_list)) if mse_list else None

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form.get('query')
    area = request.form.get('area')
    df = load_data()
    cosine_sim = build_recommender(df)

    if area:
        recommendations = recommend_travel_agents(area, df, cosine_sim, top_n=10)
    elif query:
        recommendations = recommend_travel_agents(query, df, cosine_sim, top_n=10)
    else:
        return jsonify({"error": "Please enter a search query or select an area."})

    return recommendations.to_json(orient='records')

@app.route('/evaluate', methods=['GET'])
def evaluate():
    df = load_data()
    cosine_sim = build_recommender(df)
    rmse = evaluate_recommender(df, cosine_sim)
    if rmse is not None:
        return jsonify({"rmse": f"{rmse:.2f}"})
    else:
        return jsonify({"error": "Unable to compute RMSE. Please check your data."})

if __name__ == '__main__':
    app.run(debug=True)
flas