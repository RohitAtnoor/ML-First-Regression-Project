# First step is to import the required library.
import os
import sys
from source.logger import logging
from source.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from source.components.data_transformation import DataTransformation


# Second step is to initilize the data injection configuration and files.

@dataclass
#dataclass is a function used to directly initilize the variables in a class with out the __init__ process.
#This process is used when we want to only initilize the variables, and there are no functions in the class.

class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')


# Third Step is to create a class for Data Ingection
class DataIngestion:
    # initilizing the variable.
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()    
    
    # Initiating the Data ingection process.
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts') 

        try:
            df = pd.read_csv("Notebooks/Data/gemstone.csv")  # reading the main dataset.
            logging.info('Dataset read as pandas Dataframe') 

            # copy of the main data set to another folder artifacts and new file.
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)   # saving the dataset to another folder.
            logging.info('Train test split')

            # Split the data set to Train and Test data set.
            # train_test_split will return the Train , Test data set. 
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42) 

            # save the Train data set to new folder and file. 
            # self.ingestion_config.train_data_path is the path to save the trained dataset. 
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            # save the Test data set to new folder and file.
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)

