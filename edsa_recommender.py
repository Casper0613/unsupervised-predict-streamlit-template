"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
import matplotlib
import seaborn as sns 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS



# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
ratings = pd.read_csv("resources/data/movies.csv")
movies = pd.read_csv("resources/data/ratings.csv")
df = movies.merge(ratings)
# Data Cleaning
df['genres'] = df.genres.astype(str)

df['genres'] = df['genres'].map(lambda x: x.lower().split('|'))
df['genres'] = df['genres'].apply(lambda x: " ".join(x))
st.set_page_config('centered')
# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Main Page","Recommender System","Solution Overview"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Main Page":
        st.title("Team 13 Movie Recomender System")
        st.image("resources/imgs/Main.jpg", use_column_width=True)
        st.markdown("""
        **Team : 13**
        * **Joas Sebola Tsiri:** Leader
        * **Casper Kruger:** Developed Streamlit app
        * **Nthabiseng Moloisi:** Created Notebook
        * **Rizqah Meniers:** Created Notebook
        * **Tshiamo Nthite:** Created Notebook
        """)

    if page_selection == "Solution Overview":
        st.image("resources/imgs/Solution1.jpg", width= 700 )
        st.markdown("""
        What we had to do:
        * Merge the dataset, allowing us to use both datasets.
        * Remove the pipes between genres, to be able to create graphs.
        * And convert the data type of genres to string for string handling.

        What we can see:
        * The title of the movies and their allocated ID's.
        * The genre category that each movie lies within.
        * And the ratings each movie recieved.
        """)
        st.dataframe(df)
        st.title("Rating Distribution")

        grouped = pd.DataFrame(df.groupby(['rating'])['title'].count())
        grouped.rename(columns={'title':'rating_count'}, inplace=True)
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(122)
        labels = ['0.5 Star', '1 Stars', '1.5 Stars', '2 Stars', '2.5 Stars', '3 Star', '3.5 Stars', '4 Stars', '4.5 Stars', '5 Stars']
        theme = plt.get_cmap('Blues')
        ax.set_prop_cycle("color", [theme(1. * i / len(labels))
                                 for i in range(len(labels))])
        sns.set(font_scale=1.25)
        # Create pie chart
        pie = ax.pie(grouped['rating_count'],
                 autopct='%1.1f%%',
                 shadow=True,
                 startangle=20,
                 pctdistance=1.115,
                 explode=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Most used ratings:")
            st.info("""
            * 4 Stars was the highest with 28.8%
            * 3 Stars consisted of 20.1%
            * 5 Stars consisted of 15.1%
            * 3.5 Stars consisted of 10.5%
            * 4.5 Stars consisted of 7.7%
            """)

        with col2:
            st.header("Least used ratings:")
            st.info("""
            * 0.5 Stars was the least used rating with 1.1%
            * 1.5 Stars consisted of 17%
            * 1 Stars consisted of 3.3%
            * 2.5 Stars consisted of 4.4%
            * 2 Stars consisted of 7.3%
            """)


    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
