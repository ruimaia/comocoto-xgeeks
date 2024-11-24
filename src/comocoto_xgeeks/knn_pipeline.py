import pandas as pd
import numpy as np
import re
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder

NUMERICAL_FEATURES = ["quantity", "height", "length"]

def json_to_dataframe(data: list, features: list[str]) -> pd.DataFrame:
    rows = []
    for idx, datapoint in enumerate(data):
        for c_dict in datapoint['caracteristics']:
            row = {key: c_dict[key] if key in c_dict else None for key in features}
            row["data_id"] = idx
            rows.append(row)

    df = pd.DataFrame(data=rows)

    return df

def norm_text(text):
    if text is not None:
        norm_text = str(text)
        norm_text = norm_text.lower()
        norm_text = re.sub(r'[^\w\s.\d]', '', norm_text)
        norm_text = re.sub(r'\s+', ' ', norm_text).strip()
        return norm_text
    else:
        return None
        
def extract_number(text):
    text = str(text)
    match = re.search(r'\d+\.?\d*', text)
    return match.group(0) if match else None


class BudgetEstimatorPipeline:
    def __init__(self, features, label="budget"):
        self.numerical_cols = [feature for feature in features if feature in NUMERICAL_FEATURES]
        self.categorical_cols = [feature for feature in features if feature not in self.numerical_cols]
        self.label_col = label

        self.text_cols = self.numerical_cols + self.categorical_cols

        # Define the transformers
        self.text_transformer = FunctionTransformer(self.custom_text_processing_function)
        self.numerical_converter = FunctionTransformer(self.custom_convert_to_numeric)

        self.numerical_transformer = Pipeline(steps=[
            ('norm', FunctionTransformer(self.custom_text_processing_function)),
            ('converter', FunctionTransformer(self.custom_convert_to_numeric)),
            ('scaler', StandardScaler()),
            ('imputer', KNNImputer(n_neighbors=2)),
        ])

        self.categorical_transformer = Pipeline(steps=[
            ('norm', FunctionTransformer(self.custom_text_processing_function)),
            ('imputer', SimpleImputer(strategy='constant', fill_value='na')),
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Full preprocessing pipeline with ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                #('text', self.text_transformer, self.text_cols),
                #('num_converter', self.numerical_converter, self.numerical_cols), #TODO improve
                ('num', self.numerical_transformer, self.numerical_cols),
                ('cat', self.categorical_transformer, self.categorical_cols)
            ]
        )

        self.target_transformer = FunctionTransformer(
            func=self.target_normalize_and_convert,
            validate=False
        )
        
        # Define the model pipeline: preprocessing + RandomForestRegressor
        self.model_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', KNeighborsRegressor())
        ])
        
    
    def custom_text_processing_function(self, X):
        """TODO: improve"""
        if type(X) == pd.DataFrame:
            #return X.applymap(lambda x: norm_text(x))
            return X.map(lambda x: norm_text(x))
        elif type(X) == pd.Series:
            return X.map(lambda x: norm_text(x))
    
    def custom_convert_to_numeric(self, X):
        """TODO: improve"""
        if type(X) == pd.DataFrame:
            X = X.map(lambda x: extract_number(x))
            X = X.apply(pd.to_numeric)
        elif type(X) == pd.Series:
            X = X.map(lambda x: extract_number(x))
            X = pd.to_numeric(X)
        return X
    
    def target_normalize_and_convert(self, y):
        """TODO: improve"""
        y_norm = self.custom_text_processing_function(y)
        return self.custom_convert_to_numeric(y_norm)


    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the model pipeline to the training data.
        """
        X_train = X_train[self.text_cols]
        y_train = self.target_transformer.transform(y_train)
        self.model_pipeline.fit(X_train, y_train)
    
    def predict(self, X_new: pd.DataFrame):
        """
        Make predictions on new data using the fitted pipeline.
        """
        # return self.model_pipeline.predict(X_new)
        X_new = X_new[self.text_cols]
        distances, indices = self.model_pipeline.named_steps["model"].kneighbors(self.model_pipeline.named_steps["preprocessor"].transform(X_new), n_neighbors=3)
        preds = self.model_pipeline.predict(X_new)
        return indices, preds
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        X = X[self.text_cols]
        y_pred = self.predict(X)[-1]
        y = self.target_transformer.transform(y)

        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)

        return mse, rmse, r2

    def transform(self, X: pd.DataFrame):
        """
        Transform the input data without fitting the model (for preprocessing only).
        """
        X = X[self.text_cols]
        return self.preprocessor.transform(X)
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid: dict = {"model__n_neighbors": [3,4,5]}, cv: int = 3):
        """
        Tune hyperparameters using GridSearchCV.
        """
        grid_search = GridSearchCV(self.model_pipeline, param_grid, cv=cv, n_jobs=-1, verbose=1, refit=True)
        grid_search.fit(X_train, y_train)
        self.model_pipeline = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_
