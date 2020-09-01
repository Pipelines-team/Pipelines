"""Pipelines preprocessing."""

# Authors: Sergio A. Mora Pardo <sergiomora823@gmail.com>
#          Miguel Arquez Abdala <miguel.arquez12@gmail.com>
# License: Apache-2.0

#    'Binarizer'
#    'FunctionTransformer'
#    'KBinsDiscretizer'
#    'KernelCenterer'
#    'LabelBinarizer'
#    'LabelEncoder'
#    'MultiLabelBinarizer'
#    'MinMaxScaler'
#    'MaxAbsScaler'
#    'QuantileTransformer'
#    'Normalizer'
#    'OneHotEncoder'
#    'OrdinalEncoder'
#    'PowerTransformer'
#    'RobustScaler'
#    'StandardScaler'
#    'add_dummy_feature'
#    'PolynomialFeatures'
#    'binarize'
#    'normalize'
#    'scale'
#    'robust_scale'
#    'maxabs_scale'
#    'minmax_scale'
#    'label_binarize'
#    'quantile_transform'
#    'power_transform'

import random

from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline # Construct a Pipeline from the given estimators.
from sklearn.pipeline import Pipeline # Construct a pipeline of transforms with a final estimator.
# from .Pipelines import CustomImputer - initially, based on an sklearn


class PipelineClass(Pipeline):

    """Pipeline class to simplify and automate complex Machine learning pre-processing tasks.

    Long explanation about the class.
    """

    BASE_ARGS = {
        "impute": "impute:mean",
        "scale":"scale:zstd"
    }

    def __init__(self, impute, scale, dropna, numerical_feat, 
                nominal_feat, ordinal_feat, drop_var, drop_col,
                 outlier, y_var, split):
        
        self.dataframe = None
        self.dataframe_transformed_ = None
        self.report_ = None
        self.split = split

    # This method 
    def _audit_params(self):
        """raise error if any paramer doesn't satisfy the conditions"""
        pass
    
    @property
    def dataframe(self):
        return self.dataframe

    @property
    def dataframe_transformed_(self):
        return self.dataframe_transformed_
    
    @property
    def report_(self):
        return self.report_

    def _split(self):
        # I still can't figure out the logic of the splits
        if self.split is not None:
            self.dataframe = random.sample()

    def fit(self, dataframe):
        """Execute Pipeline

        Parameters
        ----------

        dataframe: {pandas.DataFrame}
        
        
        """
        if self.impute is not None:
            # The imputer could be a class as complex as we want, but independent of the PipelineClass
            # The same logic applies to transformers, encoders, etc..
            imputer = CustomImputer() 

        return self
    
    # If we follow the scikit-learn structure we could implement this method
    def fit_transform(self):
        pass

    # For predictions/class re-usability
    def transform(self):
        pass

    def export_data(self, type='csv', path, **kwargs):
        self.dataframe_transformed_.to_csv(path, **kwargs)
        

    def get_feature_names(self):
        """Return New Df's column names
        """
        pass

    # This is only a proposal: we can separate the transformations of the main pipeline and create
    # class methods to use specfic transformation in an indepedent way
    @classmethod
    def some_transformer(cls, **kwargs):
        pass

