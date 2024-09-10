import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import math

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Utils import DataUtils
from Satisfaction_analyzer import SatAnalyzer

from data_loader import load_dataset
from data_loader import plot_cluster_analysis
from data_loader import DATA

class SatisfactionUI:   
    def __init__(self):
        st.title('TellCo User Statisfaction Analysis')

        data = DATA
        data_utils = DataUtils(data)

        cols = ['Total_sessions', 'Total_duration', 'Total_data']
        satisfacion = SatAnalyzer(cols, data, n_clusters= 3)
        data_with_score =satisfacion.score_calculator(True)

        cols = ['Avg_TCP', 'Avg_RTT', 'Avg_Throughput']
        satisfacion = SatAnalyzer(cols, data, n_clusters= 3)
        data_with_score = satisfacion.score_calculator(False)

        data_with_score['Sataisfacion_Score'] = (data_with_score['Engagement_Score'] + data_with_score['Experience_Score']) /2 

        cols = ['Engagement_Score', 'Experience_Score', 'Sataisfacion_Score']
        scores = data_with_score[cols]
        scores.sort_values(by='Sataisfacion_Score', ascending=False, inplace=True)
        

        features = ['Total_sessions', 'Total_duration', 'Total_data', 
       'Social_media', 'YouTube', 'Netflix', 'Google', 'Email', 'Gaming','Other', 
       'Avg_TCP', 'Avg_RTT',  'Avg_Throughput',
        'Engagement_Score', 'Experience_Score', 'Sataisfacion_Score']

        data_for_model = data_with_score[features]
        cols = ['Experience_Score', 'Engagement_Score']
        data_with_cluster, cluster_stats, kmeans = data_utils.cluster_analysis(cols, data_for_model, 2)

        st.markdown("<h3 style='text-align: center;'>Average Experience and Engagement score</h3>", unsafe_allow_html=True)
        plot_cluster_analysis(cols, cluster_stats)
        
        st.markdown("<h3 style='text-align: center;'>Satisfaction Score</h3>", unsafe_allow_html=True)
        fig = px.bar(data_with_cluster.groupby(by='Cluster')[['Sataisfacion_Score', 'Experience_Score']].sum())
        st.plotly_chart(fig)
        st.write('As we can see from the graph there are many satisfied customers.')




if __name__ == '__main__':
    sUi = SatisfactionUI()