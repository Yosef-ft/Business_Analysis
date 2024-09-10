import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st
from sqlalchemy import create_engine
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


sys.path.append(os.path.abspath('scripts'))

from Overview_analyzer import OverviewAnalyzer
from data_loader import load_dataset
from data_loader import pca_analyzer
from data_loader import DATA


# st.set_page_config(layout="wide")
st.title('Business analysis for an investor')
st.subheader('**TellCo company analysis**')




class UI:
    def __init__(self):
        self.data = DATA

    def main(self):
        data = self.data
        
        st.subheader('Get to know the data')
        st.text('The data that will be retrieved has been cleaned')

        activities = ['Desctibe Dataset', 'Handset Analysis', 'Correlation Analysis', 'Dimentionaly reduction Analysis']
        choice = st.selectbox("Choose the type of analysis you want to perform: ", activities)

        if choice == 'Handset Analysis':
            
            overview_analyzer = OverviewAnalyzer(data)
            Top_10_headsets = overview_analyzer.top_identifier('Handset_type', 10)
            st.markdown("<h3 style='text-align: center;'>The top ten handset types are</h3>", unsafe_allow_html=True)
            # Use st.plotly_chart for plotly figures
            fig = px.bar(Top_10_headsets)
            st.plotly_chart(fig)

        elif choice == 'Desctibe Dataset':
            st.dataframe(data.describe())

        elif choice == 'Correlation Analysis':
            cols = ['Total_data','Social_media', 'YouTube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']
            corr_matrix = data[cols].corr()
            fig = px.imshow(corr_matrix, 
                            text_auto=True,  # Add the correlation values inside the heatmap
                            aspect="auto",   # Color scale
                            labels=dict(x="Features", y="Features", color="Correlation")  # Labels for the heatmap
                        )
            fig.update_layout(
                title="Correlation Heatmap",
                title_x=0.5,  # Center the title
                xaxis_nticks=len(cols),  # Number of ticks in x-axis
                yaxis_nticks=len(cols)   # Number of ticks in y-axis
            )
            st.plotly_chart(fig)


        elif choice == 'Dimentionaly reduction Analysis':
            st.markdown("<h3 style='text-align: center;'>PCA for the total columns</h3>", unsafe_allow_html=True)
            pca_analyzer(data, True)

            st.markdown("<h3 style='text-align: center;'>PCA for the the applictions</h3>", unsafe_allow_html=True)
            pca_analyzer(data, False)



if __name__ == '__main__':
    ui = UI()

    ui.main()