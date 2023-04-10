import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = 'keywords1.csv'
model_data = pd.read_csv(data)
tfidf = TfidfVectorizer(analyzer = 'word',
                        min_df=1,
                        max_df = 0.99,
                        stop_words="english",
                        encoding = 'utf-8', 
                        token_pattern=r"(?u)\S\S+")
tfidf_encoding = tfidf.fit_transform(model_data["keywords"])

book_cosine_sim = cosine_similarity(tfidf_encoding, tfidf_encoding)

books = pd.Series(model_data['book_title'])

def recommend_books_similar_to(book_name, n=5, cosine_sim_mat=book_cosine_sim):
    # get index of the input book
    input_idx = books[books == book_name].index[0]   
    # Find top n similar books with decreasing order of similarity score
    top_n_books_idx = list(pd.Series(cosine_sim_mat[input_idx]).sort_values(ascending = False).iloc[1:n+1].index)
    # [1:6] to exclude 0 (index 0 is the input movie itself)
    recommended_books = [books[i] for i in top_n_books_idx]
    return recommended_books

def title_reformat(title):
    return title.lower().strip().replace(' ', '_')

st.title("Book Recommendatation system")
option = st.selectbox(
    'Which book is your reference point?',
    model_data['book_title'].sort_values())
nums = st.number_input("# of recommendations", 3)

if st.button('Recommend Me'):
    st.write('Books Recomended for you are:')
    option = title_reformat(option)
    st.dataframe(data=recommend_books_similar_to(option, nums))