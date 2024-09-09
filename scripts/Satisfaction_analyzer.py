import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances

from Utils import DataUtils


class SatAnalyzer:
    def __init__(self, cols, data, n_clusters):
        self.cols = cols
        data_utils = DataUtils(data)
        self.data, self.cluster_stats, self.kmeans  = data_utils.cluster_analysis(cols, data, n_clusters)

    def score_calculator(self, engagement: bool):
        '''
        This funtion calculates the engagement or experience score for a data

        Parameters:
            engagement(bool): If true calculates engagement score if false it calculates Experience score

        Returns:
            pd.DataFrame
        '''

        least_centroid = self.kmeans.cluster_centers_[0]
        user_metrics = self.data[self.cols]

        score = euclidean_distances(user_metrics, [least_centroid])

        if engagement:
            self.data['Engagement_Score'] = score
        
        else:
            self.data['Experience_Score'] = score
        
        
        return self.data