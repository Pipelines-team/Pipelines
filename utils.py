"""Submission related functions"""

import pickle
import joblib
import os

class Submission:
    """Class to create Kaggle submission
    """
    def __init__(self, submission_template, predictor, model_name):
        """init class of submission
        Args:
            submission_template (array-like): Array of predictions
            predictor (array-like): Temporal array to save predictions
            model_name (string): string with model name
        """
        self.template = submission_template
        self.predictor = predictor
        self.model_name = model_name

    def create_submission(self, predictions, save = 'submission'):
        """method to create submission
        Args:
            predictions (array-like): array of predictions
            save (str, optional): save in current working directory or sumission file. Defaults to 'submission'.
        """
        self.template[self.predictor] = predictions
        if save == 'submission':
            self.template.to_csv("submissions/" + self.model_name + "_prediction.csv", index = False)
        else:
            self.template.to_csv(os.path.dirname(__file__) + self.model_name)


class SaveModel:
    """class to save model
    """
    def __init__(self, model, model_name):
        """init method
        Args:
            model (Object): Machine Learning model to save
            model_name (String): string to name model.
        """
        self.model = model
        self.model_name = model_name

    def save_model(self):
        """method who save model
        """
        with open('data/'  + self.model_name + '.pkl') as file:
            pickle.dump(self.model, file)

class LoadModel:
    """Class to load model
    """
    def __init__(self, model_name):
        """init method
        Args:
            model_name (string): string name to load model. Needed '.pkl' or binary file.
        """
        self.model_name = model_name

    def load_model(self):
        """methos to load models
        Returns:
            Object-model: Machine Learning Model.
        """
        model_load = joblib.load(os.path.dirname(__file__) + self.model_name)
        return model_load
