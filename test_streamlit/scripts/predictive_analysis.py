import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load the dataset
df = pd.read_csv("Dataset/dairy_dataset.csv")

# Create 'Month' column
# df['Month'] = df['Date'].dt.year
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Check if 'Date' is in datetime format
if pd.api.types.is_datetime64_any_dtype(df['Date']):
    # Create 'Month' column
    df['Month'] = df['Date'].dt.month  # Correct extraction of month from 'Date'



dataset= df

df["Approx. Total Revenue(INR)"].agg(["min","mean","median","max","std","skew"]).to_frame().T

# Visualize pairwise relationships between the numerical variables
sns.pairplot(dataset[['Quantity Sold (liters/kg)', 'Price per Unit (sold)', 'Approx. Total Revenue(INR)']])
# plt.title('Pairwise Relationships')
plt.show()

# Visualize pairwise relationships between the numerical variables
sns.pairplot(dataset[['Minimum Stock Threshold (liters/kg)', 'Quantity in Stock (liters/kg)', 'Quantity (liters/kg)']])
# plt.title('Pairwise Relationships')
plt.show()

# Heatmap: Correlation Matrix
correlation_matrix = dataset[['Total Land Area (acres)', 'Number of Cows', 'Quantity (liters/kg)', 'Price per Unit', 'Total Value', 'Quantity Sold (liters/kg)', 'Price per Unit (sold)', 'Approx. Total Revenue(INR)', 'Quantity in Stock (liters/kg)', 'Reorder Quantity (liters/kg)']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


##### The correlation of the data in relation to Approx. Total Revenue (INR)


# Step 1: Select only numeric columns for correlation analysis
numeric_columns = dataset.select_dtypes(include=[np.number]).columns

# Check which columns are numeric
print("Numeric columns:", numeric_columns.tolist())

# Step 2: Compute correlations on numeric columns
correlation_matrix = dataset[numeric_columns].corr()

# Sort by a specific column, e.g., 'Approx. Total Revenue(INR)'
target_column = "Approx. Total Revenue(INR)"
if target_column in correlation_matrix.columns:
    correlation_with_target = correlation_matrix[target_column].sort_values()
    print("Correlation with 'Approx. Total Revenue(INR)':")
    print(correlation_with_target)
else:
    print(f"The target column '{target_column}' is not found among the numeric columns.")



## Spliting into Categorical and Numerical datasets

dataset.columns
dataset_num = ['Total Land Area (acres)', 'Number of Cows',
        'Quantity (liters/kg)',
       'Price per Unit', 'Total Value', 'Shelf Life (days)',
       'Quantity Sold (liters/kg)', 'Price per Unit (sold)',
       'Quantity in Stock (liters/kg)', 'Minimum Stock Threshold (liters/kg)',
       'Reorder Quantity (liters/kg)',
       'Approx. Total Revenue(INR)']

dataset_cat = ['Customer Location', 'Sales Channel', 'Product Name', 'Brand',
       'Storage Condition', 'Farm Size', 'Location']

df = df[dataset_cat]

dataset = dataset[dataset_num]



### Spiting into Independent and dependent variables

X = dataset.drop(columns=['Approx. Total Revenue(INR)']).values
y = dataset['Approx. Total Revenue(INR)'].values

## Feature Selection for importance TOP 8

from sklearn.feature_selection import SelectKBest, f_regression

# Assuming you have your feature matrix X and target vector y for regression

# Define a list of feature names in the same order as they appear in the original DataFrame X
# Replace 'feature1', 'feature2', ..., 'featureN' with the actual feature names
feature_names = ['Total Land Area (acres)', 'Number of Cows',
        'Quantity (liters/kg)',
       'Price per Unit', 'Total Value', 'Shelf Life (days)',
       'Quantity Sold (liters/kg)', 'Price per Unit (sold)',
       'Quantity in Stock (liters/kg)', 'Minimum Stock Threshold (liters/kg)',
       'Reorder Quantity (liters/kg)',
       'Approx. Total Revenue(INR)']

# Create the SelectKBest object with f_regression scoring and k=8
selector = SelectKBest(score_func=f_regression, k=8)

# Fit the selector to your data
selector.fit(X, y)

# Get the transformed feature matrix with only the selected top 5 features
X_selected = selector.transform(X)

# Get the indices of the selected features
selected_indices = selector.get_support(indices=True)

# Get the names of the selected features from the original feature_names list
selected_feature_names = [feature_names[idx] for idx in selected_indices]

print("Top selected feature names:")
print(selected_feature_names)



### Spliting into Training set and Test set

y = y.reshape(len(y), 1)

# Convert to 2D so it could fit in scaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

# This is done to scale the data either Normailization or Standardalization

# Assuming you have already defined X_train, X_test, y_train, and y_test

# Reshape y_train and y_test
y_train = y_train.ravel()
y_test = y_test.ravel()

# Now, you can proceed with fitting the model and other operations without encountering the warning.

print("The shape of X_train :",X_train.shape)
print("The shape of y_train :",y_train.shape)
print("The shape of X_test :",X_test.shape)
print("The shape of y_test :",y_test.shape)



# Applying Hyperparameter to Models

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Define models and their respective hyperparameter grids
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR()
}

param_grids = {
    'Linear Regression': {},  # No hyperparameters for Linear Regression
    'Decision Tree': {'max_depth': [None, 10, 20, 30]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'SVR': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

best_models = []
best_params = []
best_scores = []

for (model_name, model), param_grid in zip(models.items(), param_grids.values()):
    print(f'Tuning hyperparameters for {model_name}...')
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_models.append(grid_search.best_estimator_)
    best_params.append(grid_search.best_params_)
    best_scores.append(grid_search.best_score_)
    
    print(f'Best hyperparameters for {model_name}: {grid_search.best_params_}')
    print(f'Best R-squared score for {model_name}: {grid_search.best_score_}\n')



### Evaluation of model using K_Fold Cross Validation

# Reshape y

y = y.ravel()

from sklearn.model_selection import cross_val_score

# Assuming you already have best_models and X, y from the previous code
# Assuming you have fixed the shape of the target variable y using y = y.ravel()

# Perform k-fold cross-validation for each best model
k = 5  # Change k to the desired number of folds
cv_results = {}
for model_name, model in zip(models.keys(), best_models):
    cv_scores = cross_val_score(model, X, y, cv=k, scoring='r2')
    cv_results[model_name] = cv_scores

# Compute mean and standard deviation for each model's cross-validation scores
cv_mean_scores = {model_name: np.mean(scores) for model_name, scores in cv_results.items()}
cv_std_scores = {model_name: np.std(scores) for model_name, scores in cv_results.items()}

# Print the results
for model_name in cv_results.keys():
    print(f'{model_name}:')
    print(f'  Mean R-squared: {cv_mean_scores[model_name]}')
    print(f'  Standard Deviation of R-squared: {cv_std_scores[model_name]}')
    print()



## Applying Regularization

from sklearn.linear_model import LinearRegression, Ridge

# Create a Ridge Regression model with a chosen regularization strength (alpha)
ridge_model = Ridge(alpha=1.0)  # You can experiment with different alpha values
ridge_model.fit(X_train, y_train)



# Make predictions on the test set
y_pred = ridge_model.predict(X_test)



#### Evaluation of Ridge model
from sklearn.metrics import mean_squared_error, r2_score

# Calculate mean squared error and R-squared on the test set
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Round_Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")