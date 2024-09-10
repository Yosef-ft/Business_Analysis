import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import math

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Utils import DataUtils
from Experience_analyzer import ExperAnalyzer

from data_loader import load_dataset
from data_loader import plot_cluster_analysis
from data_loader import DATA


class EngagementUI:
    def __init__(self):
        st.title('TellCo User Experience Analysis')

        data = DATA
        data_utils = DataUtils(data)

        st.markdown("<h3 style='text-align: center;'>Average throughput per handset type</h3>", unsafe_allow_html=True)
        exper_analyzer = ExperAnalyzer(data)
        throughput_per_handset = exper_analyzer.Avg_metrics_per_handset('Avg_Throughput' ,data)
        fig = px.bar(throughput_per_handset.head(10))
        st.plotly_chart(fig)
        st.markdown('''**Observation** 
                    - Huawei has the most amount of Throughput. Which means that a large amount of data is being transferred efficiently, which is important for supporting high-data-demand services like video streaming, file downloads, and cloud services.''')


        st.markdown("<h3 style='text-align: center;'>Average retransmission per handset type</h3>", unsafe_allow_html=True)
        retransmission_per_handset = exper_analyzer.Avg_metrics_per_handset('Avg_TCP' ,data)
        fig = px.bar(retransmission_per_handset.head(10))
        st.plotly_chart(fig)
        st.markdown('**Observation** - At the same time the same model of Huawei that had high thourghoput has the higest retransmission, indicating a potential underlying network problems.')

        cols = ['Avg_TCP', 'Avg_RTT', 'Avg_Throughput']
        data, cluster_stats, kmeans = data_utils.cluster_analysis(cols, data, n_clusters= 3)

        st.markdown("<h3 style='text-align: center;'>Cluster analysis</h3>", unsafe_allow_html=True)
        plot_cluster_analysis(cols, cluster_stats)
        st.markdown('''
                    **Observation**
                - From the analysis using user experience data, the way I cleaned the data appears to have some effect. Since the meaning of NaN values in Avg_Tcp(retransmission) was unclear, I assumed they represented zero, indicating no retransmission or an efficient network. The clustering results reveal that a significant number of customers experience retransmissions, which could suggest low network performance. Cluster 1, which shows almost no retransmissions, suggests that very few customers face no retransmission issues.

                - From the analysis, we can see that a large amount of data has been successfully transmitted, as indicated by the Avg_Throughput. However, due to the lack of detailed information from the data collectors and the ambiguity of the NaN values, I assumed that the throughput for these NaN values is zero. This implies that the devices are connected to the network but are neither sending nor receiving any data, as shown by Cluster 1.   


                - If we assume Tellco is using Wide Area Networks (WANs), the typical RTT value ranges from 10 to 100 ms. The plots above show the RTT in seconds, with the normal range being from 0.01 to 0.1 seconds. From the clustering analysis, we can see that Clusters 0 and 1 fall within this normal range, while Cluster 2 has higher RTT values, which could indicate shortcomings in infrastructure.    
                    ''')




if __name__ == '__main__':
    eUi = EngagementUI()