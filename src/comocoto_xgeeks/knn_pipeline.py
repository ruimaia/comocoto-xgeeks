import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataProcessor:
    def __init__(self, n_neighbors=3):
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler()

    def _json_to_dataframe(self, data: dict, features: list[str]) -> pd.DataFrame:
        rows = []
        for datapoint in data:
            for c_dict in datapoint['caracteristics']:
                row = {key: c_dict[key] if key in c_dict else None for key in features}
                rows.append(row)

        df_raw = pd.DataFrame(data=rows)

        return df_raw

    def _norm_text(self, text):
        if text is not None:
            norm_text = str(text)
            norm_text = norm_text.lower()
            norm_text = re.sub(r'[^\w\s.\d]', '', norm_text)
            norm_text = re.sub(r'\s+', ' ', norm_text).strip()
        
            return norm_text

        else:
            return None
        
    def _extract_number(self, text):
        text = str(text)
        match = re.search(r'\d+\.?\d*', text)
        return match.group(0) if match else None

    def process_data(self, data: dict, selected_features: list[str], label: str = "budget") -> list[np.array]:
        """
        TODO:
            1. Distance and currency unit conversion
            2. Match similar categories (e.g. "double glaze" and "doubleglaze" -> "doubleglaze")
            3. Budget processing (e.g. 250$ each -> multiply by quantity to get total budget)
            4. Hard coded numerical labels
        """
        all_features = selected_features + [label]
        numerical_features = ["quantity", "height", "length"]
        df_raw = self._json_to_dataframe(data, all_features)
        df = df_raw.copy()

        # Process text columns
        text_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in text_cols:
            df[col] = df[col].apply(self._norm_text)

        # Convert numerical features 
        for col in numerical_features + [label]:
            df[col] = df[col].apply(self._extract_number)
            df[col] = pd.to_numeric(df[col])

        y = df[label].values
        df = df.drop(columns=label)

        # Missing values imputation
        categorical_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in categorical_cols:
            df[col] = df[col].fillna("na")

        # One-hot encode categorical features
        encoded_array = self.encoder.fit_transform(df[categorical_cols])
        df_encoded = pd.DataFrame(encoded_array, columns=self.encoder.get_feature_names_out(categorical_cols))
        df = pd.concat([df.reset_index(drop=True), df_encoded], axis=1).drop(categorical_cols, axis=1)
        del df_encoded

        # Scale numerical features
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        df[numerical_features] = self.imputer.fit_transform(df[numerical_features])

        X = df.values

        return X, y


class KNNTrainer:
    def __init__(self, param_grid: dict[str:list] = {"n_neighbors": [2,3,4,5]}, cv: int = 3):
        self.model = KNeighborsRegressor()
        self.param_grid = param_grid
        self.cv = cv
        self.grid_search = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def fit(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.grid_search = GridSearchCV(self.model, self.param_grid, cv=self.cv, n_jobs=-1, refit=True)
        self.grid_search.fit(self.X_train, self.y_train)

        self.model = self.grid_search.best_estimator_
    
    def predict(self, X):
        distances, indices = self.model.kneighbors(X, n_neighbors=3)
        preds = self.model.predict(X)
        return indices, preds
    
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        return mse, r2
    




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
    
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer

NUMERICAL_FEATURES = ["quantity", "height", "length"]

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

        #self.label_transformer = Pipeline(steps=[
        #    ('scaler', StandardScaler())
        #])
        
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
        y_train = self.target_transformer.transform(y_train)
        self.model_pipeline.fit(X_train, y_train)
    
    def predict(self, X_new: pd.DataFrame):
        """
        Make predictions on new data using the fitted pipeline.
        """
        return self.model_pipeline.predict(X_new)
        #distances, indices = self.model_pipeline.named_steps['model'].kneighbors(X_new, n_neighbors=3)
        #preds = self.model_pipeline.predict(X_new)
        #return indices, preds
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        y_pred = self.predict(X)
        y = self.target_transformer.transform(y)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        return mse, r2

    def transform(self, X: pd.DataFrame):
        """
        Transform the input data without fitting the model (for preprocessing only).
        """
        return self.preprocessor.transform(X)
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid: dict = {"model__n_neighbors": [3,4,5]}, cv: int = 3):
        """
        Tune hyperparameters using GridSearchCV.
        """
        grid_search = GridSearchCV(self.model_pipeline, param_grid, cv=cv, n_jobs=-1, verbose=1, refit=True)
        grid_search.fit(X_train, y_train)
        self.model_pipeline = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_
