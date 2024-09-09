import os
import math

import pandas as pd 
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DatabaseConn:
    '''
    This class is for creating connection to the database and fetching data
    '''
    def __init__(self):
        DB_HOST = os.getenv('DB_HOST')
        DB_PORT = os.getenv('DB_PORT')
        DB_USER = os.getenv('DB_USER')      
        DB_NAME = os.getenv('DB_NAME')      
        DB_PASSWORD = os.getenv('DB_PASSWORD')      


        self.engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}')

    def load_data_from_database(self) -> pd.DataFrame:
        '''
        This funtions creates a connection to the database and retrieves data from the database

        Returns:
        pd.DataFrame
        '''    
        with self.engine.connect() as conn:
            query = 'select * from xdr_data'
            df = pd.read_sql_query(query, conn)

        return df
    

class DataUtils:
    '''
    This class is used to clean, visualize and identify outliers, calculate PCA and perform KMeans clustering from the dataset
    '''
    def __init__(self,data: pd.DataFrame):
        self.data = data

    def data_info(self):
        '''
        Provides information about the data: 
            * provides the percentage of missing values
            * The number of missing values for each column
            * the data types of the missing values
        '''
        
        missing_values = self.data.isna().sum()
        missing_percent = self.data.isna().mean() * 100 
        data_types = self.data.dtypes

        

        info_df = pd.DataFrame({
            "Missing values" : missing_values, 
            "Missing Percentage" : missing_percent, 
            "Dtypes" : data_types
        })

        info_df = info_df[missing_percent > 0]
        info_df = info_df.sort_values(by='Missing Percentage', ascending=False)

        max_na_col = list(info_df.loc[info_df['Missing values'] == info_df['Missing values'].max()].index)
        more_than_half_na = list(info_df.loc[info_df['Missing Percentage'] > 50].index)

        print(f"**The data contains `{self.data.shape[0]}` rows and `{self.data.shape[1]}` columns.**\n\n"
            f"**The data has `{info_df.shape[0]}` missing columns.**\n\n"
            f"**The column with the maximum number of missing values is `{max_na_col}`.**\n\n"
            f"**Columns with more than 50% missing values are:**")

        # Print the list of columns with more than 50% missing values
        for column in more_than_half_na:
            print(f"- `{column}`")
            
        
        return info_df
    

    def visualize_missing_values(self):
        '''
        This method generates a heatmap to visually represent the missing values in the dataset.
        '''

        missing_cols = self.data.columns[self.data.isna().any()]

        missing_data = self.data[missing_cols]

        sns.heatmap(missing_data.T.isnull(), cbar=False)


    def visualize_outliers(self):
        '''
        This funcions helps in visualizing outliers using boxplot
        '''

        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        num_cols = len(numerical_cols)

        for i in range(0, num_cols, 5):
            fig, axes = plt.subplots(nrows = 1, ncols = 5, figsize =(20, 5))

            for j, col in enumerate(numerical_cols[i: i+5]):
                sns.boxplot(y=self.data[col], ax=axes[j])
                axes[j].set_title(col)

            plt.tight_layout()
            plt.show()



    def outlier_remover(self, columns: list) -> pd.DataFrame:
        '''
        This funtion removes all the outliers in a data using IQR technique

        Parameters: 
            columns(list): A list of columns that we don't need to remove the outlier, like unique identifiers

        Returns:
            pd.DataFrame: A dataframe without outliers
        '''

        numeric_col = self.data.select_dtypes(include='float64').columns
        fix_cols = [col for col in numeric_col if col not in columns]

        Q1 = self.data[fix_cols].quantile(0.25)
        Q3 = self.data[fix_cols].quantile(0.75)

        IQ = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQ
        upper_bound = Q3 + 1.5 * IQ

        self.data[fix_cols] = self.data[fix_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)

        return self.data  
    


    def standardize_data(self, type: str):
        '''
        This function standardizes data:       
            * This function change all columns containing ms to sec
            * changes all columns from bytes to mega-bytes

        Parameters:
            * type(str): this specifies whether the columns to standardize are volumetric or time based
        '''
        
        if type == 'time':
            columns_to_standardize = list(self.data.columns[self.data.columns.str.contains('ms')])
            self.data[columns_to_standardize] = self.data[columns_to_standardize] / 1000

            new_cols = []
            for item in columns_to_standardize:
                if ')' in item:
                    new_cols.append(item.replace('ms', 's'))
                else:
                    new_cols.append(item.replace('ms', '(s)'))
        elif type == 'volume':
            columns_to_standardize = list(self.data.columns[self.data.columns.str.contains('Bytes')])
            self.data[columns_to_standardize] = self.data[columns_to_standardize] / (1024 * 1024)
            new_cols = []
            for item in columns_to_standardize:
                if ')' in item:
                    new_cols.append(item.replace('Bytes', 'MB'))
                else:
                    new_cols.append(item.replace('Bytes', '(MB)'))            


        self.data.rename(columns=dict(zip(columns_to_standardize, new_cols)), inplace=True)

        return self.data

      
    def data_cleaning(self):
        '''
        This function cleans the data and removes outliers. This function is an aggregate of all the functions above.

        Parameters:
            None
        
        Returns:
            self.data(pd.DataFrame): a cleaned data
        '''

        self.data['Start'] = pd.to_datetime(self.data['Start'])
        self.data['End'] = pd.to_datetime(self.data['End'])

        unique_cols = ['Bearer Id', 'IMSI', 'MSISDN/Number', 'IMEI'] # columns that could be used as identifiers
        self.data = self.outlier_remover(unique_cols)

        self.data = self.standardize_data('time') # changes ms to s
        self.data = self.standardize_data('volume') # chnages bytes to Mb

        # These columns somehow are unique identifiers so filling them with other values is not a wise choice
        uni_num_cols = ['MSISDN/Number', 'Bearer Id', 'IMEI'] 
        uni_obj_cols = ['Last Location Name']

        self.data[uni_num_cols] = self.data[uni_num_cols].fillna(0) # Filling these values with zero to indicat that they are missing
        self.data[uni_obj_cols] = self.data[uni_obj_cols].fillna('undefined')

        # Although the number of null values are small compared to the dataset size I filled the null values will unkown
        # so that the analysis dosn't get affect by it.
        obj_cols = ['Handset Manufacturer', 'Handset Type']
        self.data[obj_cols] = self.data[obj_cols].fillna('unknown')


        miss_date = ['Start', 'End']
        self.data[miss_date] = self.data[miss_date].fillna(self.data[miss_date].mode(0))

        missing_IMSI = self.data['IMSI'].isna()
        self.data = self.data[~missing_IMSI]

        zero_cols = ['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (MB)', 'TCP UL Retrans. Vol (MB)']
        self.data[zero_cols] = self.data[zero_cols].fillna(0)

        mode_cols = ['Avg RTT DL (s)', 'Avg RTT UL (s)']
        mode = self.data[mode_cols].mode(0)
        self.data[mode_cols] = self.data[mode_cols].fillna(mode)


        missing_values_col = list(self.data.columns[self.data.isna().sum() != 0])
        means = self.data[missing_values_col].mean()

        # Fill missing values in these columns with the computed means
        self.data[missing_values_col] = self.data[missing_values_col].fillna(means)

        print(f">>>>>>> The data has been cleaned and outliers removed. \nThe number of null values in your data are {int(self.data.isna().sum().sum())}")

        return self.data


    def pca_analyzer(self, data: pd.DataFrame, total: bool):
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

        plt.figure(figsize=(10, 6))
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['Total_sessions'], cmap='coolwarm')
        plt.title('PCA of user behaviour')
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')


    def cluster_analysis(self, cols: list, data: pd.DataFrame, n_clusters: int):
        '''
        A function used for performing KMeans clustering and visualizing the result

        Parameter:
            cols(list): The columns that you need to perfom KMeans on
            data(pd.DataFrame): The data to perform KMeans
            n_cluster(int): The number of clusters for KMeans
        
        Returns:
            data, cluster_stats(tuple): Returns the data containing clusters and the stats for clusters dataframe         
        '''

        metrics = data[cols]
        scaler = StandardScaler()
        normalized_metrics = scaler.fit_transform(metrics)

        kmeans = KMeans(n_clusters= n_clusters, random_state=42)
        clusters = kmeans.fit_predict(normalized_metrics)

        
        data['Cluster'] = clusters       


        # Calculating cluster stats
        stats = []

        for col in cols:
            result = data.groupby('Cluster').agg({
            col: ['min', 'max', 'mean', 'sum'],
            })

            stats.append(result)

        cluster_stats = pd.concat(stats, axis =1).reset_index()

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



        return data, cluster_stats
    

    def kmean_elbow(self, cols: list, data: pd.DataFrame):
        '''
        A function used for performing KMeans clustering and visualizing the result

        Parameter:
            cols(list): The columns that you need to perfom KMeans on
            data(pd.DataFrame): The data to perform KMeans
        
        '''

        app_metrics = data[['Social_media', 'YouTube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']]
        scaler = StandardScaler()
        normalized_metrics = scaler.fit_transform(app_metrics)

        wcss = []
        k_values = range(1, 11)  

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(normalized_metrics)
            wcss.append(kmeans.inertia_)  


        plt.figure(figsize=(10, 6))
        plt.plot(k_values, wcss, marker='o')
        plt.title('Elbow Plot for K-means Clustering')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
        plt.show()        
