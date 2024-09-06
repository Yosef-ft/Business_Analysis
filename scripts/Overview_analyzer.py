import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

class OverviewAnalyzer:
    def __init__(self, data):
        self.data = data


    def top_identifier(self, column: str, no_of_columns: int) -> pd.Series:
        '''
        This function identifies the top values of a specified column and returns the data with the specified number of columns.

        Parameters:
            column (str): The column for which you want to identify the top values.
            no_of_columns (int): The number of top values to be returned.

        Returns:
            pd.Series
        '''

        Top_column = self.data[column].value_counts().sort_values(ascending = False).head(no_of_columns)
        plt.figure(figsize=(10,6))
        sns.barplot(Top_column)
        plt.xlabel(Top_column.index.name)
        plt.ylabel(f'Number of {Top_column.index.name}')
        plt.title(f'Top {no_of_columns} {Top_column.index.name}')
        plt.xticks(rotation=90)  
        plt.tight_layout()
        plt.show()      

        return Top_column  
    

    def per_user_counter(self, column, count: bool)->pd.Series:
        '''
        This function count any column per user that is per IMSI

        Parameters:
            column(str or list): the column/s you want to count or find the total
            count(bool): If count is true it counts the number per user but if false it finds the total 

        Returns:
            pd.Series
        '''

        if isinstance(column, str):

            if count:
                result = self.data.groupby(by='IMSI')[column].count().sort_values(ascending=False)

            else:
                result = self.data.groupby(by='IMSI')[column].sum().sort_values(ascending=False)
        
        else:

            if count:
                result = (self.data.groupby(by='IMSI')[column[0]].count() +
                          self.data.groupby(by='IMSI')[column[1]].count()).sort_values(ascending=False)

            else:
                result = (self.data.groupby(by='IMSI')[column[0]].sum()  + 
                          self.data.groupby(by='IMSI')[column[1]].sum()).sort_values(ascending=False)

        
        return result
    

    def univariant_plot(self, data: pd.DataFrame):
        '''
        This funtion plot violin plot for univariant data analysis

        Parameters:
            data(pd.DataFrame): data to be plotted
        '''
        data_melted = data.melt(var_name='Variable', value_name='Value')

        plt.figure(figsize=(10, 8))
        sns.violinplot(x='Variable', y='Value', data=data_melted, hue='Variable', inner='quart', palette='muted')
        plt.title('Violin Plots for All Variables')
        plt.xlabel('Variable')
        plt.ylabel('Value')     
        plt.xticks(rotation=45)
        plt.show()   
