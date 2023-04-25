import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = 'preprocessed_april25th.csv'
model_data = pd.read_csv(data, usecols=["book_title", "book_desc", "image_url", "book_title_init", "book_authors_init"])
model_data = model_data[:3000]
tfidf = TfidfVectorizer(analyzer = 'word',
                        min_df=1,
                        max_df = 0.99,
                        stop_words="english",
                        encoding = 'utf-8', 
                        token_pattern=r"(?u)\S\S+")
tfidf_encoding = tfidf.fit_transform(model_data["book_desc"])

st.title("Book Recommendatation using Descriptions")
st.markdown("🧐 How to use this recommender:\n"
            "- Describe the book (topic? story?) that you want to read about\n"
            "- Give us the number of books that you want to get\n"
            "- Click 'Recommend Me' to get your results\n"
            )
option = st.text_input("Type your query")
nums = st.number_input("# of recommendations", 3)

def book_recommend(query, num):
    results=[]
    q_vector = tfidf.transform([query])
    print("Comparable Description: ", query)
    results.append(cosine_similarity(q_vector, tfidf_encoding.toarray()))
    elem_list=[]
    for i in results[:10]:
        for elem in i[0]:
                elem_list.append(elem)

    # get a list of all titles in order of most relevant to least
    titles = []
    for i in range(num):
        titles.append("{}".format(model_data['book_title'].loc[elem_list.index(max(elem_list))]))
        elem_list.pop(elem_list.index(max(elem_list)))

    # return the dataframe in order of relevancy
    res = model_data.loc[model_data['book_title'].isin(titles), ['book_title', 'book_title_init', 'book_authors_init', 'image_url']]
    res_sorted = res.set_index('book_title').reindex(index = titles).reset_index()
    df = res_sorted.rename(columns = {'book_title_init': 'Book Title', 'book_authors_init': 'Author(s)', 'image_url': 'Cover'})
    return df[['Book Title', 'Author(s)', 'Cover']]

def path_to_image_html(path):
    return '<img src="' + path + '" width="150" >'


if st.button('Recommend Me'):
    st.write('Books Recomended for you are:')
    res = book_recommend(option, nums)
    st.markdown(
        res.to_html(escape=False, formatters=dict(Cover=path_to_image_html)),
        unsafe_allow_html=True,
        )