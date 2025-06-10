import pandas as pd
import os
from source.feature_selection.XGBoost_feature_selection import analyze_feature_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

#directories

save_dir = "D:/dev/thesis/data/feature_selection"
data_dir_train = "D:/dev/thesis/data/datasets/Edge-IIoTset/train,val and test/X_train.csv"
data_dir_test = "D:/dev/thesis/data/datasets/Edge-IIoTset/train,val and test/y_train.csv"

# Load the datasets
X_train = pd.read_csv(data_dir_train)
y_train = pd.read_csv(data_dir_test)

#Normalize numerical features
scaler = StandardScaler()
X_train[X_train.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(
    X_train.select_dtypes(include=['float64', 'int64']).values
)

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train.values.ravel())

#turn bolean columns into integers in X_train
X_train = X_train.applymap(lambda x: 1 if x is True else (0 if x is False else x))
# check colum data types of xtrain
print("X_train data types:")
print(X_train.dtypes)





#get feature names
feature_names = X_train.columns.tolist()

# Run feature importance analysis
feature_importance_df, xgb_model = analyze_feature_importance(
        X=X_train.values,
        y=y_encoded.ravel(),  # Flatten target array
        feature_names=feature_names,
        n_top_features=90,  # Show top 20 features
        save_dir=save_dir
    )

#visualize feature importance

# Sort features by importance and save to CSV
feature_importance_df.sort_values('Importance', ascending=False, inplace=True)

# Save detailed feature importance report
report_content = "Feature Importance Analysis\n"
report_content += "=" * 50 + "\n\n"
report_content += f"Total number of features: {len(feature_importance_df)}\n\n"
report_content += "Top Features by Importance:\n"
report_content += "-" * 50 + "\n"

# Add each feature and its importance score
for idx, row in feature_importance_df.iterrows():
    report_content += f"{row['Feature']}: {row['Importance']:.4f}\n"

# Save the report
with open(os.path.join(save_dir, 'feature_importance_detailed.txt'), 'w') as f:
    f.write(report_content)

# Save as CSV for easy spreadsheet viewing
feature_importance_df.to_csv(os.path.join(save_dir, 'feature_importance_scores.csv'), 
                           index=False)

print(f"Feature importance reports saved to {save_dir}")
print("Top 10 most important features:")
print(feature_importance_df.head(10))






