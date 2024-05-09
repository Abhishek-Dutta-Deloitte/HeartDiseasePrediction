import os
import pickle
import yaml
from src.logger import logger
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

# Get parent Directory path :
current_directory =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#Getting path of params.yaml
config_path = os.path.join(current_directory,"params.yaml")


def save_objects(file_path, obj):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info("Object saved successfully")
    except Exception as e:
        logger.info("Error in save_objects: {}".format(e))
        
def outlier_removal(df, num_cols):
    for column in num_cols:
        upper_limit = df[column].mean() + 2 * df[column].std()
        lower_limit = df[column].mean() - 2 * df[column].std()
        df = df[(df[column] < upper_limit) & (df[column] > lower_limit)]
    return df

def fill_empty_with_mode(df, cat_cols):
    for i in cat_cols:
        if (df[i] == '').any():
            mode_value = df[i][df[i]!=""].mode().iloc[0]
            df[i] = df[i].replace('',mode_value )
    return df


def random_search_cv(model, X_train, y_train,params):
    random_cv = RandomizedSearchCV(model, param_distributions=params, scoring="r2", cv = 5, verbose=0 )
    random_cv.fit(X_train, y_train)
    return random_cv, random_cv.best_params_, random_cv.best_score_

#Confusion Matrix

def confusion_matrix_classification_report(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    
    #Classification Report
    report = classification_report(y_test, y_pred)
    logger.info(report)
    

class FeatureClassifier:
    def __init__(self,df, target_column):
        self.df = df
        self.target_column = target_column
    
    def get_ordinal_columns_mapping(self,columns):
        """
        This function is used to get the mapping of ordinal columns.
        Each key is named as 'ColumnName_Map' and contains the unique values for that column.
        
        This method takes a list of column names as input and returns a dictionary where each key is a column name and its corresponding value is a 
        list of unique values in that column sorted by the mean of the target_column for each unique value. This method is used to get mappings for 
        ordinal columns (columns with ordered categorical data).
        """
        columns_mapping = {}
        
        for col in columns:
            sorted_groups = self.df.groupby(col)[self.target_column].mean().sort_values().index.tolist()
            key_name = f"{col}"
            columns_mapping[key_name] = sorted_groups
        
        return columns_mapping
        

        
    def ordinal_onehot_numerical_divide(self):
        """
        This function is used to divide the categorical into ordinal and one-hot columns and numerical columns.
        
        Explanation: In this function, the ordinal_onehot_numerical_divide function makes the distinction based on the standard deviation of the mean target value for each category. 
        If the standard deviation is greater than 10% of the overall mean target value, the column is considered ordinal. Otherwise, it's considered suitable for one-hot encoding. 
        This is a more data-driven approach and it assumes that if there's a significant variation in the target variable across categories, those categories might have a meaningful order.
        """
        one_hot_cols = []
        ordinal_cols = []
        num_cols = []
        #Overall mean
        mean = self.df[self.target_column].mean()
        thereshold_percentage = 0.1
        threshold_value = mean * thereshold_percentage
        try:
            for column in self.df.columns:
                if column != self.target_column and self.df[column].dtype == 'object':
                    df_column = self.df[[column, self.target_column]].groupby(column).mean().reset_index()
                    standard_dev = df_column[self.target_column].std()
                    if standard_dev > threshold_value:
                        ordinal_cols.append(column)
                    else:
                        one_hot_cols.append(column)
                else:
                    num_cols.append(column)
            
            logger.info("Outliers removed!!!")

            #Get Mappingsd for ordinal columns:
            ordinal_columns_mapping = self.get_ordinal_columns_mapping(ordinal_cols)
            one_hot_column_mapping = self.get_ordinal_columns_mapping(one_hot_cols)
            return (one_hot_cols, ordinal_cols, num_cols, ordinal_columns_mapping, one_hot_column_mapping)
                 

        except Exception as e:
            logger.info(e)

def save_objects(file_path, obj):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info("Object saved successfully")
    except Exception as e:
        logger.info("Error in save_objects: {}".format(e))
    

def load_obj(file_path):
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info("Object loaded successfully")
        return obj
    except Exception as e:
        logger.info("Error in load_obj: {}".format(e))
    
from sklearn.metrics import accuracy_score


def random_search_cv(model, X_train, y_train, params):
    random_cv = RandomizedSearchCV(model, param_distributions=params, scoring="accuracy", cv = 5, verbose=0 )
    random_cv.fit(X_train, y_train)
    return random_cv, random_cv.best_params_, random_cv.best_score_

def evaluate_model(X_train, y_train, X_test, y_test, models):
    report = {}
    
    # config_path = "../params.yaml"
    #Load yaml file:x
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        for i in range(len(models)):
            model = list(models.values())[i]
            model_flag = list(models.keys())[i]
            # model.fit(X_train, y_train)
            
            params = config[model_flag]
            model, model.best_params_, model.best_score_ = random_search_cv(model, X_train, y_train, params)
            
            y_pred = model.predict(X_test)

            test_model_score = accuracy_score(y_test, y_pred)
            logger.info('\n====================================================================================\n')
            logger.info(f"The confusion matrix and classification report for the model: {model_flag} is:")
            confusion_matrix_classification_report(y_test, y_pred)
            logger.info('\n====================================================================================\n')


            logger.info('\n====================================================================================\n')
            logger.info(f"The best parameters for the model{model_flag} are {model.best_params_}")
            logger.info('\n====================================================================================\n')


            report[list(models.keys())[i]] =  {"score": test_model_score, "best_params": model.best_params_}
            logger.info(f"Model: {list(models.keys())[i]}, Accuracy score: {test_model_score}")
        logger.info("Model evaluation complete")
        return report

    except Exception as e:
        logger.info("Error in evaluate_model: {}".format(e))
