import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataProcessor:
    def __init__(self, n_neighbors=3):
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler()

    def _json_to_dataframe(data: dict, features: list[str]) -> pd.DataFrame:
        rows = []
        for datapoint in data:
            for c_dict in datapoint['caracteristics']:
                row = {key: c_dict[key] if key in c_dict else None for key in features}
                rows.append(row)

        df_raw = pd.DataFrame(data=rows)

        return df_raw

    def _norm_text(text):
        if text is not None:
            norm_text = str(text)
            norm_text = norm_text.lower()
            norm_text = re.sub(r'[^\w\s.\d]', '', norm_text)
            norm_text = re.sub(r'\s+', ' ', norm_text).strip()
        
            return norm_text

        else:
            return None
        
    def _extract_number(text):
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
        numerical_features = ["quantity", "height", "distance"]
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

        X = df.values

        return X, y


class KNNTrainer:
    def __init__(self, param_grid: list[int] = [3,4,5], cv: int = 3):
        self.model = KNeighborsRegressor()
        self.param_grid = param_grid
        self.cv = cv
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def fit(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def predict(self, X):
        self.model.predict(X)