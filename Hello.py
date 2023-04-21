import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to BookRec 👋")

st.sidebar.success("Select a search method.")

st.markdown(
    """
    BookRec is a book recommender tool built to help you find the 
    right book for your need (even when you are not sure of what you want).
    \n
    **👈 Select a method from the sidebar** to start searching!
    ### Search methods:
    - With Keywords: Get your recommendation(s) using keywords
    - With Queries: Get your recommendation(s) using a description of your ideal book
    - With Similar Book: Get your recommendation(s) using a reference 
"""
)