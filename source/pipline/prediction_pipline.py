import sys
import os
from source.exception import CustomException
from source.logger import logging
from source.utils import load_object
import pandas as pd

# creating the class for predicting the new data. 
class PredictPipeline:
    def __init__(self):
        pass
    
    # function returns the predicted value.
    # features is the input columns details.
    def predict(self,features):
        try:
            # getting the preprocessor and model pickle files path. 
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            # opening the preprocessor and model file 
            # load_object is function in utilis to open the files. 
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            # transforming the new data. 
            data_scaled=preprocessor.transform(features)

            # predicting the price. 
            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)


# data or the columns input data. 
class CustomData:
    # initilizing the variables. 
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity
    
    # storing the variables in the dictionary and converting to Dataframe. 
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)