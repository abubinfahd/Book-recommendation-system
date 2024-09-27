# Book-recommendation-system
This project focuses on building a popularity and collaborative filtering-based book recommender system. It uses users' book ratings to recommend books based on user preferences and rating patterns.

## Popularity-Based Book Recommender System

This section implements a popularity-based recommendation system that suggests the most highly rated books with a minimum number of ratings.

### Steps:

1. **Merging Ratings with Book Information**  
   Merge the `ratings` dataset with the `books` dataset on the `ISBN` to include book details.
   ```python
   ratings_with_name = ratings.merge(books, on='ISBN')
   ratings_with_name.head()
  
2. **Calculating Number of Ratings per Book**
Group the data by book title and count the number of ratings each book received.
```python
num_ratings_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_ratings_df.rename(columns={'Book-Rating':'num_ratings'}, inplace = True)

```
3. **Calculating Average Rating per Book**
Calculate the average rating for each book.
```python
avg_rating_df = ratings_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_rating'}, inplace=True)

```
4. **Merging Number of Ratings and Average Rating**
Combine the number of ratings and average rating into one DataFrame.
```python
popular_df = num_ratings_df.merge(avg_rating_df, on='Book-Title')

```
5. **Filtering Books with Minimum Number of Ratings**
Filter books with at least 250 ratings and sort them by average rating in descending order. Keep only the top 50 books.
```python
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)

```
6. **Final Popular Books DataFrame**
Merge this filtered list with the books DataFrame to include additional details like author and image, and remove duplicates.
```python
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')\
    [['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]
popular_df

```
This approach recommends books that are popular based on the number of ratings and average rating, ensuring that only highly rated books with significant user engagement are suggested.

## Collaborative Filtering-Based Book Recommender System

This section implements a collaborative filtering recommendation system. It recommends books based on similarities between users' preferences and ratings using the cosine similarity metric.

### Steps:

1. **Filtering Users with Sufficient Ratings**  
   Filter users who have rated more than 200 books.
   ```python
   x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
   good_users = x[x].index
   filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(good_users)]
   filtered_rating.head()
2. **Filtering Books with Sufficient Ratings**
Filter books that have received at least 50 ratings.
```python
y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
good_ratings_book = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(good_ratings_book)]
final_ratings.drop_duplicates()

```
3. **Creating a Pivot Table**
Create a pivot table with Book-Title as the index and User-ID as the columns, where the values represent the ratings.
```python
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)
pt.head()

```
4. **Calculating Cosine Similarity**
Use cosine similarity to compute the similarity score between books based on user ratings.
```python
from sklearn.metrics.pairwise import cosine_similarity
similarity_score = cosine_similarity(pt)
similarity_score.shape

```
5. **Recommendation Function**
Define a function to recommend books similar to a given book title based on cosine similarity.
```python
def recommend(book_name):
    if book_name not in pt.index:
        print(f"Book '{book_name}' not found in the dataset.")
        return
    # fetch index
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]

    for i in similar_items:
        print(pt.index[i[0]])

```
This collaborative filtering approach uses user ratings to recommend similar books, leveraging cosine similarity to measure the closeness between books in terms of user preferences.
