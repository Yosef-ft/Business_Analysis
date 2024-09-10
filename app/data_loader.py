import math
import os
import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_dataset():

    LIVE_DB_USER = st.secrets['LIVE_DB_USER']
    LIVE_DB_PASSWORD = st.secrets['LIVE_DB_PASSWORD']
    LIVE_DB_HOST = st.secrets['LIVE_DB_HOST']
    LIVE_DB_NAME = st.secrets['LIVE_DB_NAME']    

    host = f'postgresql://{LIVE_DB_USER}:{LIVE_DB_PASSWORD}@{LIVE_DB_HOST}/{LIVE_DB_NAME}'
    engine = create_engine(host)
    
    @st.cache_resource
    def fetch_data():
        with engine.connect() as conn:
            query = 'SELECT * FROM xdr_data'
            df = pd.read_sql_query(query, conn)
        return df

    return fetch_data()

DATA = load_dataset()
DATA = DATA.iloc[1:]

def pca_analyzer(data: pd.DataFrame, total: bool):
    '''
    This function calculates the PCA  and visualizes the results.

    Parameters:
        data(pd.DataFrame): The input data
        total(bool): This is whether to perform PCA on aggregate columns or individual columns

    Returns:
        matplotlib.pyplot plot: A plot showing the PCA results.
    '''

    scaler = StandardScaler()

    if total:
        cols = ['Total_duration', 'Total_sessions', 'Total_data']
    else: 
        cols = ['Social_media', 'YouTube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']
    scaled_data = scaler.fit_transform(data[cols])

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    fig = px.scatter(x=pca_data[:, 0], 
                    y=pca_data[:, 1], 
                    color=data['Total_sessions'], 
                    labels={'x': 'First principal component', 'y': 'Second principal component'},
                    title='PCA of user behaviour')
    
    st.plotly_chart(fig)


def plot_cluster_analysis(cols, cluster_stats):
        
        ncols = 3
        nrows = math.ceil(len(cols) / ncols)

        if len(cols) == 3:
            st.markdown("<h3 style='text-align: center;'>Average Values per Cluster</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='text-align: center;'>Application engagement per Cluster</h3>", unsafe_allow_html=True)
        
        
        ncols = 3
        nrows = math.ceil(len(cols) / ncols)
        
        # Create the subplots with the correct number of rows and columns
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 5 * nrows))
        
        axes = axes.flatten()
        
        # Loop over the columns and plot each one
        for i, item in enumerate(cols):
            sns.barplot(x='Cluster', y=cluster_stats[item]['mean'], data=cluster_stats, ax=axes[i])
            axes[i].set_title(f'Average {item} per Cluster') 

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])   

        st.pyplot(fig)