
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Normalize numerical features in all dataframes
def normalize_features(df):
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns].values)
    return df

# Encode target labels for all target dataframes
def encode_labels(y_df):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_df.values.ravel())
    return y_encoded