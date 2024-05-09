import os
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from src.logger import logger
from src.exceptions import CustomException

from dataclasses import dataclass
import yaml
from src.utils import *



@dataclass
class ModelTrainerConfig:

    # Get parent Directory path :
    current_directory =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  

    # Define artifacts path
    artifact_path = os.path.join(current_directory, "artifacts")

    #Define model.pkl path:
    trained_model_path = os.path.join(artifact_path, "model.pkl")
    
class InitiateModelTraining:
    def __init__(self) :
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        """
        This function is used to initiate the model training process.
        """
        logger.info("Initiating model training process...")
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )

            models = {
                'SVC': SVC(),
                'DecisionTree':DecisionTreeClassifier(random_state=42),
                'RandomForest':RandomForestClassifier(random_state=42),
                'GradientBoostingClassification':GradientBoostingClassifier(random_state=42),
                'BaggingClassification' : BaggingClassifier(random_state=42)
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models=models)

            logger.info('\n====================================================================================\n')
            logger.info(f'Model Report : {model_report}')
            logger.info('\n====================================================================================\n')

            # best_model_score = max(sorted(model_report.values()))
            # best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            # logger.info('\n====================================================================================\n')
            # logger.info(f'Best Model: {best_model_name} ## Best Model Score : {best_model_score}')
            # logger.info('\n====================================================================================\n')



            # Get the model names and their scores
            model_scores = {model_name: model_info['score'] for model_name, model_info in model_report.items()}

            # Find the model with the best score
            best_model_name = max(model_scores, key=model_scores.get)
            best_model_score = model_scores[best_model_name]

            # Get the best parameters for the best model
            best_model_params = model_report[best_model_name]['best_params']
            best_model = models[best_model_name]
            
            # Create a new instance of the model using the best parameters
            best_model = models[best_model_name].set_params(**best_model_params)

            # Fit the model with the best parameters
            best_model.fit(X_train, y_train)

            #Save object:
            save_objects(self.model_trainer_config.trained_model_path, best_model)
            logger.info("The model is saved successfully to {}".format(self.model_trainer_config.trained_model_path))
        except Exception as e:
            logger.error("Error initiating model training process", e)
            raise CustomException(e)
