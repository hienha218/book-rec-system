import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def rm_punc(text):
    return re.sub(r'\W+|_', ' ', text.lower())

data = 'keywords1.csv'
model_data = pd.read_csv(data)
tfidf = TfidfVectorizer(analyzer = 'word',
                        min_df=1,
                        max_df = 0.99,
                        stop_words="english",
                        encoding = 'utf-8', 
                        token_pattern=r"(?u)\S\S+")
tfidf_encoding = tfidf.fit_transform(model_data["keywords"])

tfidf_df = pd.DataFrame(tfidf_encoding.toarray(), index=model_data["book_title"], columns=tfidf.get_feature_names_out())
# Sort maximum tf-idf total score
tfidf_df["total"]= tfidf_df.sum(axis=1)
tfidf_df = tfidf_df.sort_values("total", ascending=False)
del tfidf_df["total"]

# Leave first few words containing years
tfidf_df_preview = tfidf_df.iloc[:,25:].copy()

#Fix 100 to 0
tfidf_df_preview = tfidf_df_preview.stack().reset_index()
tfidf_df_preview = tfidf_df_preview.rename(columns={0:'tfidf', 'book_title': 'book','level_1': 'term'})
tfidf_df_preview = tfidf_df_preview.sort_values(by=['book','tfidf'], ascending=[True,False]).groupby(['book']).head(10)

def process_word_matrix(word_vec):
    # Remove underscores in terms
    word_vec.term = word_vec.term.str.replace('_',' ')

    # Remove terms with zero tfidf score
    word_vec = word_vec[word_vec.tfidf > 0]
    
    return word_vec

tfidf_vec = process_word_matrix(tfidf_df_preview.copy())    

def findBookWithSimilarTerm(term):
  res = set()
  for i in range(len(tfidf_vec)):
    if tfidf_vec.iloc[i]['term'] == term or tfidf_vec.iloc[i]['term'] in term or term in tfidf_vec.iloc[i]['term']:
      res.add(tfidf_vec.iloc[i]['book'])
  return res

def findBookWithListofSimilarTerms(listTerm):
  result = set()
  for term in listTerm:
    tempList = findBookWithSimilarTerm(term)
    result = result.union(tempList)
  return list(result)

st.title("Book Recommendatation system")
option = st.text_input("Type your query")
nums = st.number_input("# of recommendations", 3)

def book_recommend(query, num):
    results = findBookWithListofSimilarTerms(query)
    r = []
    for i in range(num):
        r.append("{}".format(model_data[model_data['book_title']==results[i]].book_title))
    return r

if st.button('Recommend Me'):
    st.write('Books Recomended for you are:')
    option = rm_punc(option).split(" ")
    st.dataframe(data=book_recommend(option, nums))