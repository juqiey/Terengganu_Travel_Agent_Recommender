from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np
import os

app = Flask(__name__)

# Load the dataset
def load_data():
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_travel_agent.csv')
    df = pd.read_csv(dataset_path)
    
    # Data cleaning
    df['rating'] = df['rating'].replace('no review yet', '0')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
    
    # Fill NaN values
    df['category'] = df['category'].fillna('')
    df['area'] = df['area'].fillna('')
    df['type'] = df['type'].fillna('')
    df['validity'] = df['validity'].fillna('')
    
    # Create content field for TF-IDF
    df['content'] = (df['name'] + ' ' + 
                    df['category'] + ' ' + 
                    df['area'] + ' ' + 
                    df['type'] + ' ' + 
                    df['validity'] + ' Rating:' + 
                    df['rating'].astype(str))
    
    return df

# Build the recommender system
def build_recommender(df):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Create TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(df['content'])
    
    # Compute cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return vectorizer, cosine_sim

# Recommend travel agents
def recommend_travel_agents(input_query, df, cosine_sim, vectorizer=None, top_n=10):
    # Case 1: If input is a rating (numeric)
    try:
        input_rating = float(input_query)
        filtered_df = df[df['rating'] >= input_rating].sort_values('rating', ascending=False)
        
        if filtered_df.empty:
            return pd.DataFrame()  # Return empty DataFrame if no matches
            
        return filtered_df[['name', 'category', 'area', 'type', 'validity', 'rating']].head(top_n)
    
    # Case 2: If input is not a rating
    except ValueError:
        # Case 2a: If input is an area name
        areas = ['Kuala Terengganu', 'Kuala Nerus', 'Setiu', 'Besut', 'Marang', 'Kemaman', 'Dungun']
        if input_query in areas:
            filtered_df = df[df['area'] == input_query].sort_values('rating', ascending=False)
            
            if filtered_df.empty:
                return pd.DataFrame()  # Return empty DataFrame if no matches
                
            return filtered_df[['name', 'category', 'area', 'type', 'validity', 'rating']].head(top_n)
        
        # Case 2b: General search query
        else:
            # Create a new TF-IDF matrix for the input query
            if vectorizer is None:
                vectorizer = TfidfVectorizer(stop_words='english')
                vectorizer.fit(df['content'])
                
            # Transform input query to TF-IDF vector
            try:
                query_vec = vectorizer.transform([input_query])
                
                # Get similarity scores
                sim_scores = cosine_similarity(query_vec, vectorizer.transform(df['content'])).flatten()
                
                # Get top N similar items
                top_indices = sim_scores.argsort()[-top_n:][::-1]
                
                # Check if we have any reasonable matches (similarity > 0)
                if len(top_indices) > 0 and max(sim_scores) > 0:
                    return df.iloc[top_indices][['name', 'category', 'area', 'type', 'validity', 'rating']]
                else:
                    return pd.DataFrame()  # Return empty DataFrame if no matches
                    
            except Exception as e:
                print(f"Error in recommendation: {e}")
                return pd.DataFrame()  # Return empty DataFrame on error

# Evaluate the recommender system
def evaluate_recommender(df, cosine_sim):
    # Split data into train and test sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Get indices of test agents
    test_indices = pd.Series(test.index, index=test['name']).drop_duplicates()
    
    mse_list = []
    
    for agent_name in test['name'].unique():
        if agent_name in test_indices:
            idx = test_indices[agent_name]
            
            # Get similarity scores for this agent
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get the 5 most similar agents (excluding itself)
            sim_indices = [i[0] for i in sim_scores[1:6]]
            
            if len(sim_indices) > 0:
                # Get true ratings
                true_ratings = df.iloc[sim_indices]['rating']
                
                # Calculate predicted rating (average of similar agents)
                if not true_ratings.empty:
                    predicted_rating = true_ratings.mean()
                    actual_rating = df.loc[idx, 'rating']
                    
                    # Calculate MSE
                    mse_list.append((actual_rating - predicted_rating) ** 2)
    
    # Calculate RMSE
    if mse_list:
        return np.sqrt(np.mean(mse_list))
    else:
        return None

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get query parameters
    query = request.form.get('query', '')
    area = request.form.get('area', '')
    
    # Load data and build recommender
    try:
        df = load_data()
        vectorizer, cosine_sim = build_recommender(df)
        
        # Get recommendations based on area or query
        if area:
            recommendations = recommend_travel_agents(area, df, cosine_sim, vectorizer, top_n=10)
        elif query:
            recommendations = recommend_travel_agents(query, df, cosine_sim, vectorizer, top_n=10)
        else:
            return jsonify({"error": "Please enter a search query or select an area."})
        
        # Return results
        if recommendations.empty:
            return jsonify([])  # Return empty array if no recommendations
        else:
            return recommendations.to_json(orient='records')
            
    except Exception as e:
        print(f"Error in /recommend route: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"})

@app.route('/evaluate', methods=['GET'])
def evaluate():
    try:
        # Load data and build recommender
        df = load_data()
        _, cosine_sim = build_recommender(df)
        
        # Evaluate recommender
        rmse = evaluate_recommender(df, cosine_sim)
        
        if rmse is not None:
            return jsonify({"rmse": f"{rmse:.2f}"})
        else:
            return jsonify({"error": "Unable to compute RMSE. Please check your data."})
            
    except Exception as e:
        print(f"Error in /evaluate route: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)