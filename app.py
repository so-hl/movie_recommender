from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
import time

app = Flask(__name__)

# Global variables for models
ncf_model = None
movies_content = None
tfidf_matrix = None
tfidf_vectorizer = None
user2idx = None
movie2idx = None
idx2movie = None
num_users = None
num_movies = None
device = None
ratings = None
movies = None

# NCF Model Definition (must match your training)
class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50, hidden_dims=[128, 64, 32]):
        super(NCF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)
        
        layers = []
        input_dim = embedding_dim * 2
        
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(0.2))
            input_dim = dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        self.output_activation = nn.Sigmoid()
        
    def forward(self, user_ids, movie_ids):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        
        x = torch.cat([user_emb, movie_emb], dim=-1)
        x = self.mlp(x).squeeze()
        x = self.output_activation(x) * 4.5 + 0.5
        
        return x

def load_models():
    """Load all saved models and data"""
    global ncf_model, movies_content, tfidf_matrix, tfidf_vectorizer
    global user2idx, movie2idx, idx2movie, num_users, num_movies, device
    global ratings, movies
    
    try:
        print("Loading models...")
        
        # Set device
        device = torch.device('cpu')
        
        # Check if data files exist
        if not os.path.exists('./data/ratings.csv'):
            raise FileNotFoundError("ratings.csv not found in ./data/")
        if not os.path.exists('./data/movies.csv'):
            raise FileNotFoundError("movies.csv not found in ./data/")
            
        # Load original data
        ratings = pd.read_csv('./data/ratings.csv')
        movies = pd.read_csv('./data/movies.csv')
        print(f"Loaded {len(ratings)} ratings and {len(movies)} movies")
        
        # Check if model files exist
        if not os.path.exists('./models/ncf_complete.pth'):
            raise FileNotFoundError("ncf_complete.pth not found in ./models/")
            
        # Load NCF components with weights_only=False for PyTorch 2.6+
        checkpoint = torch.load('./models/ncf_complete.pth', map_location=device, weights_only=False)
        user2idx = checkpoint['user2idx']
        movie2idx = checkpoint['movie2idx']
        idx2movie = checkpoint['idx2movie']
        num_users = checkpoint['num_users']
        num_movies = checkpoint['num_movies']
        
        # Initialize and load NCF model
        ncf_model = NCF(num_users, num_movies).to(device)
        ncf_model.load_state_dict(checkpoint['model_state_dict'])
        ncf_model.eval()
        print("NCF model loaded")
        
        # Load content-based components
        if not os.path.exists('./models/movies_content.csv'):
            raise FileNotFoundError("movies_content.csv not found in ./models/")
        if not os.path.exists('./models/tfidf_matrix.npz'):
            raise FileNotFoundError("tfidf_matrix.npz not found in ./models/")
            
        movies_content = pd.read_csv('./models/movies_content.csv')
        tfidf_matrix = sparse.load_npz('./models/tfidf_matrix.npz')
        
        with open('./models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        print("All models loaded successfully!")
        
    except Exception as e:
        print(f"ERROR loading models: {str(e)}")
        raise

# Recommendation functions - OPTIMIZED VERSION
def get_ncf_recommendations(user_id, top_k=10):
    """Get NCF recommendations for a user - OPTIMIZED VERSION"""
    if user_id not in user2idx:
        return None
    
    user_idx = user2idx[user_id]
    
    # Get movies the user hasn't rated
    user_ratings = ratings[ratings['userId'] == user_id]
    rated_movies = set(user_ratings['movieId'].values)
    
    # Instead of scoring ALL movies, sample a subset
    all_movie_ids = list(movie2idx.keys())
    unrated_movies = [m for m in all_movie_ids if m not in rated_movies]
    
    # Sample a reasonable number of movies to score (e.g., 5000 instead of 60000+)
    sample_size = min(5000, len(unrated_movies))
    sampled_movies = random.sample(unrated_movies, sample_size)
    
    # Create tensors only for sampled movies
    sampled_movie_indices = torch.tensor([movie2idx[m] for m in sampled_movies], dtype=torch.long).to(device)
    user_indices = torch.full((len(sampled_movies),), user_idx, dtype=torch.long).to(device)
    
    # Batch predictions for better performance
    batch_size = 1000
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(sampled_movies), batch_size):
            batch_users = user_indices[i:i+batch_size]
            batch_movies = sampled_movie_indices[i:i+batch_size]
            batch_predictions = ncf_model(batch_users, batch_movies)
            all_predictions.extend(batch_predictions.cpu().numpy())
    
    # Get top-k movies from the sample
    predictions_tensor = torch.tensor(all_predictions)
    top_values, top_indices = torch.topk(predictions_tensor, min(top_k, len(predictions_tensor)))
    
    # Get the actual movie IDs
    top_movie_ids = [sampled_movies[idx] for idx in top_indices]
    
    recommendations = []
    for movie_id in top_movie_ids:
        if movie_id in movies_content['movieId'].values:
            movie_info = movies_content[movies_content['movieId'] == movie_id].iloc[0]
            recommendations.append({
                'movieId': int(movie_id),
                'title': movie_info['title'],
                'genres': movie_info['genres']
            })
    
    return recommendations

def get_content_recommendations(user_id, top_k=10):
    """Get content-based recommendations - OPTIMIZED VERSION"""
    user_ratings = ratings[ratings['userId'] == user_id]
    
    if user_ratings.empty:
        return None
    
    # Get liked movies (4+ rating)
    liked_movies = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].values
    
    if len(liked_movies) == 0:
        liked_movies = user_ratings.nlargest(5, 'rating')['movieId'].values
    
    # Limit to most recent 10 liked movies to speed up computation
    liked_movies = liked_movies[:10]
    
    # Get indices of liked movies
    liked_indices = []
    for movie_id in liked_movies:
        if movie_id in movies_content['movieId'].values:
            idx = movies_content[movies_content['movieId'] == movie_id].index[0]
            liked_indices.append(idx)
    
    if not liked_indices:
        return []
    
    # Compute similarities in one batch operation
    liked_vectors = tfidf_matrix[liked_indices]
    
    # Instead of computing similarity with ALL movies, sample a subset
    n_movies = tfidf_matrix.shape[0]
    sample_size = min(5000, n_movies)  # Sample 5000 movies max
    sample_indices = random.sample(range(n_movies), sample_size)
    
    # Compute similarities only with sampled movies
    sample_vectors = tfidf_matrix[sample_indices]
    similarities = cosine_similarity(liked_vectors, sample_vectors)
    
    # Average similarities across all liked movies
    avg_similarities = similarities.mean(axis=0)
    
    # Get already rated movies to exclude
    rated_movies = set(user_ratings['movieId'].values)
    
    # Create list of (index, score) for sorting
    movie_scores = []
    for i, sim_score in enumerate(avg_similarities):
        actual_idx = sample_indices[i]
        movie_id = movies_content.iloc[actual_idx]['movieId']
        if movie_id not in rated_movies:
            movie_scores.append((actual_idx, sim_score))
    
    # Sort by similarity score
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    recommendations = []
    for idx, score in movie_scores[:top_k]:
        movie_info = movies_content.iloc[idx]
        recommendations.append({
            'movieId': int(movie_info['movieId']),
            'title': movie_info['title'],
            'genres': movie_info['genres']
        })
    
    return recommendations

def get_hybrid_recommendations(user_id, top_k=10, alpha=0.6):
    """Get hybrid recommendations - OPTIMIZED VERSION"""
    # Time each method
    start = time.time()
    ncf_recs = get_ncf_recommendations(user_id, top_k*2)
    ncf_time = time.time() - start
    print(f"NCF took {ncf_time:.2f} seconds")
    
    start = time.time()
    content_recs = get_content_recommendations(user_id, top_k*2)
    content_time = time.time() - start
    print(f"Content-based took {content_time:.2f} seconds")
    
    if ncf_recs is None and content_recs is None:
        return []
    elif ncf_recs is None:
        return content_recs[:top_k]
    elif content_recs is None:
        return ncf_recs[:top_k]
    
    # Score combination
    scores = {}
    
    # NCF scores
    for i, rec in enumerate(ncf_recs):
        movie_id = rec['movieId']
        scores[movie_id] = alpha * (len(ncf_recs) - i) / len(ncf_recs)
    
    # Content scores
    for i, rec in enumerate(content_recs):
        movie_id = rec['movieId']
        if movie_id not in scores:
            scores[movie_id] = 0
        scores[movie_id] += (1 - alpha) * (len(content_recs) - i) / len(content_recs)
    
    # Get top movies
    top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    recommendations = []
    for movie_id, _ in top_movies:
        movie_info = movies_content[movies_content['movieId'] == movie_id].iloc[0]
        recommendations.append({
            'movieId': int(movie_id),
            'title': movie_info['title'],
            'genres': movie_info['genres']
        })
    
    return recommendations

def find_similar_movies(movie_title, top_k=10):
    """Find similar movies based on content"""
    matches = movies_content[movies_content['title'].str.contains(movie_title, case=False, na=False)]
    
    if matches.empty:
        return None
    
    movie_info = matches.iloc[0]
    idx = matches.index[0]
    
    # Compute similarities on-demand
    movie_vector = tfidf_matrix[idx:idx+1]
    similarities = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    
    # Get top similar movies (excluding itself)
    similar_indices = similarities.argsort()[::-1][1:top_k+1]
    
    similar_movies = []
    for idx in similar_indices:
        movie = movies_content.iloc[idx]
        similar_movies.append({
            'movieId': int(movie['movieId']),
            'title': movie['title'],
            'genres': movie['genres']
        })
    
    return {
        'query_movie': movie_info['title'],
        'similar_movies': similar_movies
    }

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_id = int(data['user_id'])
        method = data['method']
        alpha = float(data.get('alpha', 0.6))
        
        if method == 'ncf':
            recommendations = get_ncf_recommendations(user_id)
        elif method == 'content':
            recommendations = get_content_recommendations(user_id)
        else:  # hybrid
            recommendations = get_hybrid_recommendations(user_id, alpha=alpha)
        
        if recommendations is None:
            return jsonify({'error': 'User not found or no recommendations available'}), 404
        
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/similar', methods=['POST'])
def similar():
    try:
        data = request.json
        movie_title = data['movie_title']
        
        result = find_similar_movies(movie_title)
        
        if result is None:
            return jsonify({'error': 'Movie not found'}), 404
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user_stats/<int:user_id>')
def user_stats(user_id):
    try:
        user_ratings = ratings[ratings['userId'] == user_id]
        
        if user_ratings.empty:
            return jsonify({'error': 'User not found'}), 404
        
        stats = {
            'user_id': user_id,
            'total_ratings': len(user_ratings),
            'average_rating': round(user_ratings['rating'].mean(), 2),
            'rating_distribution': user_ratings['rating'].value_counts().sort_index().to_dict(),
            'top_rated_movies': []
        }
        
        # Get top rated movies
        top_movies = user_ratings.nlargest(5, 'rating').merge(movies, on='movieId')
        for _, row in top_movies.iterrows():
            stats['top_rated_movies'].append({
                'title': row['title'],
                'rating': float(row['rating']),
                'genres': row['genres']
            })
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/random_users')
def random_users():
    """Get random user IDs for demo purposes"""
    try:
        # Get users with reasonable number of ratings
        user_counts = ratings['userId'].value_counts()
        active_users = user_counts[user_counts > 20].index.tolist()
        
        # Sample 10 random users
        sample_users = random.sample(active_users[:100], min(10, len(active_users)))
        
        return jsonify({'users': sample_users})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test')
def test():
    return jsonify({'status': 'Flask is running!', 'models_loaded': ncf_model is not None})

if __name__ == '__main__':
    # Load models when starting the server
    load_models()
    
    # Run the app
    app.run(debug=True, port=5000, use_reloader=False)