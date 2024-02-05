import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def rename_columns(df):
    df = df.rename(columns={'sales': 'department', 'salary': 'salary_level'})
    return df


def encode_features(df):
    df['department'] = df['department'].map({
        'sales': 0, 'technical': 1, 'support': 2, 'IT': 3, 'product_mng': 4,
        'marketing': 5, 'RandD': 6, 'accounting': 7, 'hr': 8, 'management': 9
    })

    df['salary_level'] = df['salary_level'].map({'low': 0, 'medium': 1, 'high': 2})
    return df


def create_train_test_split(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler().set_output(transform="pandas")
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler


