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

plt.figure(figsize=(15, 10))  # Increased figure size
plt.bar(range(len(feature_importance_df)), feature_importance_df['Importance'])
plt.xticks(range(len(feature_importance_df)), 
          feature_importance_df['Feature'], 
          rotation=90,  # Vertical text
          ha='center',  # Center alignment
          fontsize=8)   # Smaller font size
plt.title('Feature Importance', pad=20)  # Add padding to title
plt.xlabel('Features', labelpad=10)
plt.ylabel('Importance Score')

# Adjust layout to prevent label cutoff
plt.subplots_adjust(bottom=0.3)  # More space at bottom for labels
plt.tight_layout()

# Save with higher DPI for better quality
plt.savefig(os.path.join(save_dir, 'feature_importance_plot.png'), 
            dpi=300,
            bbox_inches='tight')  # Ensure no labels are cut off







