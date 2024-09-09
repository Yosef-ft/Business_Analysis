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
    

    def per_user_counter(self,customer_ID,  column, count: bool)->pd.Series:
        '''
        This function count any column per user that is per IMSI

        Parameters:
            customer_ID(str): the ID used to identify the customer
            column(str or list): the column/s you want to count or find the total
            count(bool): If count is true it counts the number per user but if false it finds the total 

        Returns:
            pd.Series
        '''

        if isinstance(column, str):

            if count:
                result = self.data.groupby(by=customer_ID)[column].count().sort_values(ascending=False)

            else:
                result = self.data.groupby(by=customer_ID)[column].sum().sort_values(ascending=False)
        
        else:

            if count:
                result = (self.data.groupby(by=customer_ID)[column[0]].count() +
                          self.data.groupby(by=customer_ID)[column[1]].count()).sort_values(ascending=False)

            else:
                result = (self.data.groupby(by=customer_ID)[column[0]].sum()  + 
                          self.data.groupby(by=customer_ID)[column[1]].sum()).sort_values(ascending=False)

        
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


    def user_behaviour_Id(self, ID: str, data: pd.DataFrame):
        '''
        This function classifies the given data by user and calculates the total data for each apps

        Parameter:
            ID(str): the Id of the data, It could be IMSI or MSISDN

        Returns:
            user_behaviour(pd.DataFrame): The user behaviour for each apps per customer ID
        '''

        xdr_per_user = self.per_user_counter(ID, 'Bearer Id', count=True)
        session_dur = self.per_user_counter(ID,'Dur. (s)', count=False)
        total_dl_user = self.per_user_counter(ID,'Total DL (MB)', count=False)
        total_ul_user = self.per_user_counter(ID,'Total UL (MB)', count=False)

        cols = ['Social Media UL (MB)', 'Social Media DL (MB)']
        social_media = self.per_user_counter(ID,cols, count=False)

        cols = ['Youtube UL (MB)', 'Youtube DL (MB)']
        YouTube = self.per_user_counter(ID,cols, count=False)

        cols = ['Netflix UL (MB)', 'Netflix DL (MB)']
        Netflix = self.per_user_counter(ID,cols, count=False)

        cols = ['Google UL (MB)', 'Google DL (MB)']
        Google = self.per_user_counter(ID,cols, count=False)

        cols = ['Email UL (MB)', 'Email DL (MB)']
        Email = self.per_user_counter(ID,cols, count=False)

        cols = ['Gaming UL (MB)', 'Gaming DL (MB)']
        Gaming = self.per_user_counter(ID,cols, count=False)

        cols = ['Other UL (MB)', 'Other DL (MB)']
        Other = self.per_user_counter(ID,cols, count=False)

        user_behaviour_by_ID = pd.concat([xdr_per_user, session_dur, total_dl_user,
                                    total_ul_user, social_media, YouTube,
                                    Netflix, Google, Email,
                                    Gaming, Other], axis=1)

        user_behaviour_by_ID.columns = ['Total_sessions', 'Total_duration', 'Total_Dl', 'Total_Ul', 'Social_media', 
                                'YouTube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']

        user_behaviour_by_ID['Total_data'] = user_behaviour_by_ID['Total_Dl'] + user_behaviour_by_ID['Total_Ul']        


        return user_behaviour_by_ID