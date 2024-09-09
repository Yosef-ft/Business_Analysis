import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ExperAnalyzer:
    def __init__(self, data):
        self.data = data

    def top_bottom_frequent(self):
        '''
        This function calculate displays the top, bottom and most frequent of some colmns
        '''
        
        cols = ['Avg_TCP', 'Avg_Throughput', 'Avg_RTT']

        top = self.data[cols].head(10)
        bottom = self.data[cols].head(10)

        most_frequent_TCP = self.data['Avg_TCP'].sort_values(ascending = False)
        most_frequent_Throughput = self.data['Avg_Throughput'].sort_values(ascending = False)
        most_frequent_RTT = self.data['Avg_RTT'].sort_values(ascending = False)

        print('>>>>>>>> TOP <<<<<<<<<<<<')
        print(top, '\n\n\n')

        print('>>>>>>>> Bottom <<<<<<<<<<<<')
        print(bottom, '\n\n\n')

        print('>>>>>>>> most frequent TCP <<<<<<<<<<<<')
        print(most_frequent_TCP, '\n\n\n')

        print('>>>>>>>> most frequent Throughput <<<<<<<<<<<<')
        print(most_frequent_Throughput, '\n\n\n')

        print('>>>>>>>> most frequent RTT <<<<<<<<<<<<')
        print(most_frequent_RTT, '\n\n\n')    


    def Avg_metrics_per_handset(self,metrics: str ,data: pd.DataFrame):
        '''
        This function extracts the average throughput per handse

        Parameters:
            data(pd.DataFrame): input data
            metrics(str): metics like `Avg_Throughput` or `Avg_TCP` for average retransmission

        Returns 
            pd.DataFrmae
        '''   

        return data.loc[data['Handset_Count'] == 1].groupby(by='Handset_type')[metrics].sum().sort_values(ascending = False)