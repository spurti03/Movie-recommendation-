import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample movie dataset
data = {
    'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump'],
    'genre': ['Drama', 'Crime', 'Action', 'Crime', 'Drama'],
    'director': ['Frank Darabont', 'Francis Ford Coppola', 'Christopher Nolan', 'Quentin Tarantino', 'Robert Zemeckis'],
    'description': ['Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.', 
                    'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.', 
                    'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.', 
                    'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.', 
                    'The presidencies of Kennedy and Johnson, the events of Vietnam, Watergate, and other historical events unfold through the perspective of an Alabama man with an IQ of 75.']
}

# Create a DataFrame
movies = pd.DataFrame(data)

# Vectorize the description column
tfidf = TfidfVectorizer(stop_words='english')
movies['description'] = movies['description'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['description'])

# Compute similarity scores
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on similarity scores
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Get recommendations for a movie
movie_title = 'The Shawshank Redemption'
recommended_movies = get_recommendations(movie_title)
print(f"Recommended movies for '{movie_title}':")
print(recommended_movies)
