import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import Image, display, HTML

# Load datasets
books = pd.read_csv('E:\\data science\\books reccoo proje\\Books1.csv')
users = pd.read_csv('E:\\data science\\books reccoo proje\\Users.csv')
ratings = pd.read_csv('E:\\data science\\books reccoo proje\\Ratings.csv')

# Data cleaning and processing (as in your original code)
books.dropna(inplace=True)
valid_years = books['Year-Of-Publication'].astype(str).str.isnumeric()
books = books[valid_years]
books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
books['Publication_Date'] = pd.to_datetime(books['Year-Of-Publication'], format='%Y', errors='coerce')
books.drop(columns=['Year-Of-Publication'], inplace=True)
books['Year-Of-Publication'] = books['Publication_Date'].dt.year
books = books[~books['Year-Of-Publication'].isin([2037, 2026, 2030, 2050, 2038])]

# Create user ratings DataFrame
ratings_books_name = ratings.merge(books, on='ISBN')
numer_rating = ratings_books_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
numer_rating.rename(columns={'Book-Rating':'Total_number_rating'}, inplace=True)
avg_rating = ratings_books_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating.rename(columns={'Book-Rating':'Total_avg_rating'}, inplace=True)

# Merge ratings
popular_df = numer_rating.merge(avg_rating, on='Book-Title')
popular_df = popular_df[popular_df['Total_number_rating'] >= 250].sort_values('Total_avg_rating', ascending=False).head(50)
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Image-URL-M', 'Total_number_rating', 'Total_avg_rating']]

# Prepare the pivot table for recommendations
pt = ratings_books_name.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)
similarityscore = cosine_similarity(pt)

# Recommendation function
def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarityscore[index])), key=lambda x: x[1], reverse=True)[1:6]
    
    recommendations = []
    for i in similar_items:
        book_title = pt.index[i[0]]
        image_url = books.loc[books['Book-Title'] == book_title, 'Image-URL-M'].values[0]
        google_search_url = f"https://www.google.com/search?q={book_title.replace(' ', '+')}"
        recommendations.append((book_title, image_url, google_search_url))
    
    return recommendations

# Example of using the recommend function
recommended_books = recommend('Whispers')

# Display recommendations with images and Google search links
for title, img_url, google_search_url in recommended_books:
    display(HTML(f"<a href='{google_search_url}' target='_blank'><img src='{img_url}' width='100' height='150'></a><br>{title}"))
