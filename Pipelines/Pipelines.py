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

from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline # Construct a Pipeline from the given estimators.
from sklearn.pipeline import Pipeline # Construct a pipeline of transforms with a final estimator.

