import streamlit as st
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
import pandas as pd
import requests

# Load data
data = Dataset.load_builtin('ml-100k')
data_me = pd.read_csv('movie.csv')

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize model
algo = SVD()
algo.fit(trainset)

# Function to search movie by name using TMDB API
def search_movie_by_name(movie_name):
    api_key = 'c7ec19ffdd3279641fb606d19ceb9bb1'
    url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_name}'
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            return results[0]
    return None

# Streamlit UI
st.set_page_config(page_title="Collaborative Recommendation", page_icon=":clapper:", layout="wide")

# Header
st.title('Collaborative Movie Recommendation')

# Login section
st.sidebar.subheader('Login')
user_id = st.sidebar.text_input('Enter User ID')
login_button = st.sidebar.button('Login')

# Main content
if login_button:
    authenticated_user_ids = [str(uid) for uid in range(1, 944)]
    if user_id in authenticated_user_ids:
        st.sidebar.success('Login Successful!')
        
        already_rated = trainset.ur[trainset.to_inner_uid(user_id)]
        
        st.subheader(f"Welcome, user {user_id}!")
        
        st.write("Movies already rated/watched:")
        for movie_id, rating in already_rated[:5]:
            movie_name = int(trainset.to_raw_iid(movie_id)) + 1
            naam = data_me.loc[data_me['movieId'] == movie_name, 'title'].values
            if naam:
                st.image('https://via.placeholder.com/200x300.png?text=Movie+Poster', caption=naam[0], width=200)
            else:
                st.write(f"Movie ID {movie_name} not found")
        
        all_movie_ids = list(trainset.all_items())
        movies_not_rated_by_user = [movie_id for movie_id in all_movie_ids if movie_id not in trainset.ur[trainset.to_inner_uid(user_id)]]
        
        user_predictions = [algo.predict(user_id, movie_id) for movie_id in movies_not_rated_by_user]
        user_predictions.sort(key=lambda x: x.est, reverse=True)
        
        st.subheader(f"Top 10 recommended movies for user {user_id}:")
        # Create a grid layout for the recommended movies
        col1, col2, col3 = st.columns(3)
        for i, prediction in enumerate(user_predictions[:30]):
            movie_id = prediction.iid
            movie_name = int(trainset.to_raw_iid(movie_id)) + 1
            naam = data_me.loc[data_me['movieId'] == movie_name, 'title'].values
            if naam:
                if i < 10:
                    with col1, st.expander(f"{i+1}. {naam[0]} (Predicted Rating: {prediction.est:.2f})"):
                        st.image('https://via.placeholder.com/200x300.png?text=Movie+Poster', caption=naam[0], width=200)
                elif i < 20:
                    with col2, st.expander(f"{i+1}. {naam[0]} (Predicted Rating: {prediction.est:.2f})"):
                        st.image('https://via.placeholder.com/200x300.png?text=Movie+Poster', caption=naam[0], width=200)
                else:
                    with col3, st.expander(f"{i+1}. {naam[0]} (Predicted Rating: {prediction.est:.2f})"):
                        st.image('https://via.placeholder.com/200x300.png?text=Movie+Poster', caption=naam[0], width=200)
            else:
                st.write(f"Movie ID {movie_name} not found")
    else:
        st.sidebar.error('Invalid User ID. Please try again.')