"""
ml/random_forest.py
Integrasi output Fuzzy (Mamdani & Sugeno) sebagai fitur tambahan
untuk Random Forest Regressor pada dataset Zomato.
"""

import time
import numpy as np
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from fuzzy.mamdani import mamdani_predict
from fuzzy.sugeno  import sugeno_predict


def generate_fuzzy_features(df, sample_size=None):
    """Tambahkan kolom fuzzy_mamdani dan fuzzy_sugeno ke dataframe."""
    df_s = df.sample(n=sample_size, random_state=42).copy() if sample_size else df.copy()
    mamdani_scores, sugeno_scores = [], []

    for _, row in df_s.iterrows():
        mamdani_scores.append(
            mamdani_predict(row['Average Cost for two'], row['Votes'],
                            row['Price range'], row['Has Table booking'],
                            row['Has Online delivery'])['score']
        )
        sugeno_scores.append(
            sugeno_predict(row['Average Cost for two'], row['Votes'],
                           row['Price range'], row['Has Table booking'],
                           row['Has Online delivery'])['score']
        )

    df_s['fuzzy_mamdani'] = mamdani_scores
    df_s['fuzzy_sugeno']  = sugeno_scores
    return df_s


def train_and_evaluate(df_fuzz):
    """
    Latih dua model Random Forest:
    1. RF tanpa fitur fuzzy
    2. RF dengan fitur fuzzy tambahan
    """
    base_features  = ['Average Cost for two', 'Votes', 'Price range',
                      'Has Table booking', 'Has Online delivery']
    fuzzy_features = base_features + ['fuzzy_mamdani', 'fuzzy_sugeno']
    target = 'Aggregate rating'

    df_fuzz = df_fuzz.dropna(subset=[target])
    results = {}

    for tag, features, label in [
        ('rf_raw',   base_features,  'RF tanpa Fuzzy'),
        ('rf_fuzzy', fuzzy_features, 'RF + Fuzzy Features'),
    ]:
        X = df_fuzz[features].values
        y = df_fuzz[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        t0  = time.time()
        clf = RandomForestRegressor(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        rt  = round(time.time() - t0, 3)

        pred = clf.predict(X_test)
        mae  = round(float(np.mean(np.abs(y_test - pred))), 4)
        mse  = round(float(np.mean((y_test - pred) ** 2)), 4)
        rmse = round(float(np.sqrt(mse)), 4)

        results[tag] = {
            'label' : label,
            'mae'   : mae,
            'mse'   : mse,
            'rmse'  : rmse,
            'runtime': rt,
            'pred'  : pred,
            'actual': y_test,
        }

    return results
