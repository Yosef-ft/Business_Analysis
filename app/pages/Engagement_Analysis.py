import os
import sys

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import math

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from Utils import DataUtils
st.write(os.getcwd())
from data_loader import load_dataset
from data_loader import plot_cluster_analysis
from data_loader import DATA


class EngagementUI:
    def __init__(self):
        st.title('TellCo User Engagement Analysis')

        data = DATA
        st.write("The top 5 most engaged users", data.head())


        data_utils = DataUtils(data)
        cols = ['Total_sessions', 'Total_duration', 'Total_data']
        apps = ['Social_media', 'YouTube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']
        data, cluster_stats, kmeans = data_utils.cluster_analysis(cols, data=data, n_clusters=3)
        data, cluster_stats_apps, kmeans = data_utils.cluster_analysis(apps, data=data, n_clusters=3)


        plot_cluster_analysis(cols, cluster_stats)
        plot_cluster_analysis(apps, cluster_stats_apps)


        st.markdown("<h3 style='text-align: center;'>Analysis of application to show which is more engaging app</h3>", unsafe_allow_html=True)
        app_metrics = data[['Social_media', 'YouTube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']]

        fig = px.bar(app_metrics.max().sort_values())

        st.plotly_chart(fig)







if __name__ == '__main__':
    ui = EngagementUI()