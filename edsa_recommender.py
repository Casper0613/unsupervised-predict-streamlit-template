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
from re import X
from tkinter import Y
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

        


        if st.button('Show raw data'):
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
        
        plt.tight_layout()
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Most used ratings:")
            st.info("""
            * **4** - Stars was the highest with 28.8%
            * **3** - Stars consisted of 20.1%
            * **5** - Stars consisted of 15.1%
            * **3.5** - Stars consisted of 10.5%
            * **4.5** - Stars consisted of 7.7%
            """)

        with col2:
            st.header("Least used ratings:")
            st.info("""
            * **0.5** - Stars was the least used rating with 1.1%
            * **1.5** - Stars consisted of 17%
            * **1** - Stars consisted of 3.3%
            * **2.5** - Stars consisted of 4.4%
            * **2** - Stars consisted of 7.3%
            """)

        st.title('Genres popularity:')   
        st.image("resources/imgs/genres.png", width= 850)
        
        st.header('What the graph shows:')
        st.info("""
        * Throughout the years the amount of times that movies was rated differs.
        * The graph also shows that certain Genres where rated more than others.
        * This could mean that the ratings poeple gave to movies was a factor of the type of movies that came out each year.
        * The top 4 most rated genres are Drama, Comedy, Action and Thriller.
        * Which means people were more intrested in rating movies with these types of Genres than any other type.
        """)

        st.title('Model we used:')
        
        st.image("resources/imgs/model.jpeg", width= 700 )
        st.markdown('From here we decided wich models to use. Our decision came from each models advantages, disadvantages and how they performed against each other.')
        
        st.title('Collaborative recommender systems we used :')
        st.image('resources/imgs/SVD.png')
        st.header('(SVD) Singular Value Decomposition :')
        

        st.latex(r'''
        A = U \sum V^T
        ''')

        col3, col4 = st.columns(2)

        with col3:
            st.header('Advantages')
            st.info("""
            * Can be apploed to non-square matrices
            * Making the observation have the largest variance
            * SVD can be utilized to sully forth pseudo-inverses.
            """)
        with col4:
            st.header('Disadvantages')
            st.info("""
            * Computing is very slow
            * Computationally expensive
            * Requires care when dealing with missing data
            """)

        st.header('(KNN) K-Nearest Neighbor :')

        st.latex(r'''
        r_{ij} = \sum_k Similaries(u_i,u_k)r_{kj} / {number-of-ratings}
        ''')

        col5, col6 = st.columns(2)
        with col5:
            st.header('Advantages')
            st.info("""
            * It is simple to implement.
            * It is robust to the noisy training data.
            * It can be more effective if the training data is large.
            """)
        with col6:
            st.header('Disadvantages')
            st.info("""
            * Always needs to determine the value of K which may be complex some time.
            * The computation cost is high because of calculating the distance between the data points for all the training samples.
            """)

            



    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
