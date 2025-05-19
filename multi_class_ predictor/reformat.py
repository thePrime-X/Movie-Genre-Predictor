import pandas as pd
import ast

# Load the original dataset
df = pd.read_csv('movies-and-shows.csv')

# Drop unnecessary columns (imdb-related columns)
columns_to_drop = ['imdb_id', 'imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Filter out rows where 'genres' column is empty (contains [])
df = df[df['genres'].apply(lambda x: x != '[]')]

# Reformat the 'genres' column
def reformat_genres(genre_string):
    # Convert the string from a list format to an actual list
    genres = ast.literal_eval(genre_string)
    # Join the list elements into a single comma-separated string
    return ', '.join(genres)

df['genres'] = df['genres'].apply(reformat_genres)

# Save the reformatted dataset as a new CSV file
df.to_csv('reformatted_movies.csv', index=False)

print("Reformatted movies saved as 'reformatted_movies.csv'.")
