import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = 'keywords3.csv'
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
    print(book_name)
    # get index of the input book
    input_idx = books[books == book_name].index[0]   
    # Find top n similar books with decreasing order of similarity score
    # [1:6] to exclude 0 (index 0 is the input book itself)
    top_n_books_idx = list(pd.Series(cosine_sim_mat[input_idx]).sort_values(ascending = False).iloc[1:n+1].index)
    recommended_books = [books[i] for i in top_n_books_idx]

    res = model_data.loc[model_data['book_title'].isin(recommended_books), ['book_title', 'book_title_init', 'book_authors_init', 'image_url']]
    res_sorted = res.set_index('book_title').reindex(index = recommended_books).reset_index()
    df = res_sorted.rename(columns = {'book_title_init': 'Book Title', 'book_authors_init': 'Author(s)', 'image_url': 'Cover'})
    return df[['Book Title', 'Author(s)', 'Cover']]


st.title("Book Recommendation using Similar Book")
st.markdown("🧐 How to use this recommender:\n"
            "- Choose a book in our database\n"
            "- Give us the number of books that you want to get\n"
            "- Click 'Recommend Me' to get your results\n"
            )
option_choice = st.selectbox(
    'Which book is your reference point?',
    model_data['book_title_init'].sort_values())
nums = st.number_input("# of recommendations", 3)

def path_to_image_html(path):
    return '<img src="' + path + '" width="150" >'

if st.button('Recommend Me'):
    st.write('Books Recomended for you are:')
    option = model_data.loc[model_data['book_title_init']==option_choice, 'book_title'].values[0]
    res = recommend_books_similar_to(option, nums)
    st.markdown(
        res.to_html(escape=False, formatters=dict(Cover=path_to_image_html)),
        unsafe_allow_html=True,
        )