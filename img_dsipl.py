import streamlit as st
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
import pandas as pd
import requests

# Load the MovieLens dataset (download it from http://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
data = Dataset.load_builtin('ml-100k')
data_me = pd.read_csv('movie.csv')

# Split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Use the SVD algorithm
algo = SVD()

# Train the model on the training set
algo.fit(trainset)

# Function to search for a movie by name
def search_movie_by_name(movie_name):
    api_key = 'c7ec19ffdd3279641fb606d19ceb9bb1'  # Replace 'YOUR_API_KEY' with your actual TMDB API key
    url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_name}'
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            # Return the first result
            return results[0]
    return None

# Streamlit UI
st.title('Movie Recommendation System')

# Login Page
st.subheader('Login')
user_id = st.text_input('Enter User ID')

# Authenticate user
if st.button('Login'):
    authenticated_user_ids = [str(uid) for uid in range(1, 944)]  # Assuming user IDs range from 1 to 943
    if user_id in authenticated_user_ids:
        st.success('Login Successful!')
        
        # Get movies already rated by the user
        already_rated = trainset.ur[trainset.to_inner_uid(user_id)]
        
        # Home Page
        st.subheader(f"Welcome, user {user_id}!")
        
        # Display movies already rated/watched with images
        st.write("Movies already rated/watched:")
        for movie_id, rating in already_rated[:5]:  # Displaying only 5 movies for brevity
            movie_name = int(trainset.to_raw_iid(movie_id)) + 1
            naam = data_me.loc[data_me['movieId'] == movie_name, 'title'].values
            if naam:
                st.write(f"{naam[0]} (Rating: {rating})")
                result = search_movie_by_name(naam[0])
                print(naam[0])
                if result:
                    poster_path = result.get('poster_path')
                    if poster_path:
                        poster_url = f"https://image.tmdb.org/t/p/w185/{poster_path}"
                        st.image(poster_url, caption=naam[0], width=100)
                    else:
                        st.write(f"No poster found for {naam[0]}")
            else:
                st.write(f"Movie ID {movie_name} not found")
        
        # Get the list of all movie IDs
        all_movie_ids = list(trainset.all_items())
        
        # Get the list of movie IDs that the user with the specified user ID has not rated
        movies_not_rated_by_user = [movie_id for movie_id in all_movie_ids if movie_id not in trainset.ur[trainset.to_inner_uid(user_id)]]
        
        # Make predictions for the unrated movies
        user_predictions = [algo.predict(user_id, movie_id) for movie_id in movies_not_rated_by_user]
        
        # Sort the predictions by predicted rating in descending order
        user_predictions.sort(key=lambda x: x.est, reverse=True)
        
        # Display the top 10 recommended movies with images
        st.subheader(f"Top 10 recommended movies for user {user_id}:")
        for i, prediction in enumerate(user_predictions[:10]):
            movie_id = prediction.iid
            movie_name = int(trainset.to_raw_iid(movie_id)) + 1
            naam = data_me.loc[data_me['movieId'] == movie_name, 'title'].values
            if naam:
                st.write(f"{i+1}. {naam[0]} (Predicted Rating: {prediction.est:.2f})")
                result = search_movie_by_name(naam[0])
                if result:
                    poster_path = result.get('poster_path')
                    if poster_path:
                        poster_url = f"https://image.tmdb.org/t/p/w185/{poster_path}"
                        st.image(poster_url, caption=naam[0], width=100)
                    else:
                        st.write(f"No poster found for {naam[0]}")
            else:
                st.write(f"Movie ID {movie_name} not found")
    else:
        st.error('Invalid User ID. Please try again.')
