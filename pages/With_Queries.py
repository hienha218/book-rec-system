import streamlit as st
import pandas as pd
import numpy as np
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

st.title("Book Recommendatation system")
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

    r = []
    for i in range(num):
        r.append("{}".format(model_data['book_title'].loc[elem_list.index(max(elem_list)):elem_list.index(max(elem_list))]))
        elem_list.pop(elem_list.index(max(elem_list)))
    return r

if st.button('Recommend Me'):
     st.write('Books Recomended for you are:')
     st.dataframe(data=book_recommend(option, nums))