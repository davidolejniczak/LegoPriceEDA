import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression, LassoCV

df = pd.read_csv('data/LegoTrainingData.csv')

'''Piece Count vs Retail Price'''
x = df['num_parts']
y = df['retail_price']

coefficients = np.polyfit(x, y, deg=1)  # Linear trendline
trendline = np.poly1d(coefficients)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.4, label='Data Points', color='red')
plt.plot(x, trendline(x), color='blue', linewidth=2, label=f'Trendline: y={coefficients[0]:.2f}x + {coefficients[1]:.2f}')
plt.title('Piece Count vs. Retail Price of LEGO Sets with Trendline', fontsize=16)
plt.xlabel('Piece Count', fontsize=12)
plt.ylabel('Retail Price (USD)', fontsize=12)
plt.legend()
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Charts/PieceCountVsRetailPrice.png')
plt.close()

'''Plot 2: ROI Heatmap by Category'''
themeRoi = df.groupby('category')['return'].mean().reset_index()
themeRoi = themeRoi.sort_values(by='return', ascending=False)

plt.figure(figsize=(12, 8))
sns.heatmap(themeRoi[['return']].set_index(themeRoi['category']), 
            annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Average ROI'})
plt.title('ROI Heatmap by Category', fontsize=16)
plt.ylabel('Category', fontsize=12)
plt.xlabel('ROI', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('Charts/ROIHeatmapbyCategory.png')
plt.close()

'''Plot 3: Barplot for Average ROI'''
plt.figure(figsize=(14, 8))
sns.barplot(x='return',data=themeRoi, palette="viridis", hue='category')
plt.title('Average ROI by Category', fontsize=16)
plt.xlabel('Average ROI', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.tight_layout()
plt.savefig('Charts/AverageROIbyCategory.png')
plt.close()

'''Least Square Error Regression'''
dfencoded = pd.get_dummies(df, columns=['category'], drop_first=True)

# Select relevant columns
dfmodel = dfencoded[['num_parts', 'num_figs', 'retail_price', 
                     'num_unique_figs', 'set_rating', 'pop_price', 
                     'return']]
X = dfmodel.drop(['return'], axis=1)
y = dfmodel['return']

# Debug missing data
print("Columns with NaN values:", X.columns[X.isna().any()].tolist())
print("Number of NaN values per column:\n", X.isna().sum())

# Handle missing values
# Split data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Model fitting
model = LinearRegression()
model.fit(Xtrain, ytrain)

ytrainpred = model.predict(Xtrain)
ytestpred = model.predict(Xtest)

print("Training R2 Score:", r2_score(ytrain, ytrainpred))
print("Training MAE Score:", mean_absolute_error(ytrain, ytrainpred))
print("Test R2 Score:", r2_score(ytest, ytestpred))
print("Test MAE Score:", mean_absolute_error(ytest, ytestpred))

plt.figure(figsize=(10, 6))
plt.scatter(ytest, ytestpred, alpha=0.5, color='orange')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs Predicted (Test Set)")
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Charts/LSEPrediction.png')
plt.close()

import numpy as np

# Define the range of x values for the trend line
x = np.linspace(ytest.min(), ytest.max(), 100)

# Compute the trend line using the linear regression model coefficients
coefficients = np.polyfit(ytest, ytestpred, 1)  # Fit a 1-degree polynomial (straight line)
trendline = np.poly1d(coefficients)  # Create a polynomial function

# Plot the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(ytest, ytestpred, alpha=0.5, color='orange', label="Data Points")
plt.plot(x, trendline(x), color='blue', linewidth=2, label=f'Trendline: y={coefficients[0]:.2f}x + {coefficients[1]:.2f}')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs Predicted (Test Set)")
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save or display the plot
plt.savefig('Charts/regression_plot_with_trendline.png')
plt.close()


'''Cross-Validation Regression'''
crossValR2Scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("Cross-Validation R2 Score:", crossValR2Scores.mean())

mae = make_scorer(mean_absolute_error, greater_is_better=False)
crossValMaeScores = cross_val_score(model, X, y, cv=5, scoring=mae)
print("Cross-Validation MAE Score:", -crossValMaeScores.mean())

predicted_y = cross_val_predict(model, X, y, cv=5)

plt.figure(figsize=(10, 6))
plt.scatter(y, predicted_y, alpha=0.5, color='purple')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Cross-Validation Predictions")
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Charts/CrossValidationPredictions.png')
plt.close()

'''Lasso Regression'''
lasso = LassoCV(cv=5, random_state=42).fit(Xtrain, ytrain)

ytrainpred = lasso.predict(Xtrain)
ytestpred = lasso.predict(Xtest)

print("Training R2 Score (Lasso):", r2_score(ytrain, ytrainpred))
print("Training MAE Score (Lasso):", mean_absolute_error(ytrain, ytrainpred))
print("Test R2 Score (Lasso):", r2_score(ytest, ytestpred))
print("Test MAE Score (Lasso):", mean_absolute_error(ytest, ytestpred))

plt.figure(figsize=(10, 6))
plt.scatter(ytest, ytestpred, alpha=0.5, color='green')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Lasso Regression Predictions (Test Set)")
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Charts/LassoPrediction.png')
plt.close()

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lasso.coef_
}).sort_values(by="Coefficient", ascending=False)
print(coefficients)

import numpy as np

# Define the range of x values for the trend line
x = np.linspace(ytest.min(), ytest.max(), 100)

# Compute the trend line using the Lasso model
coefficients = np.polyfit(ytest, ytestpred, 1)  # Fit a 1-degree polynomial (straight line)
trendline = np.poly1d(coefficients)  # Create a polynomial function

# Plot the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(ytest, ytestpred, alpha=0.5, color='green', label="Data Points")
plt.plot(x, trendline(x), color='blue', linewidth=2, label=f'Trendline: y={coefficients[0]:.2f}x + {coefficients[1]:.2f}')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Lasso Regression Predictions (Test Set)")
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save or display the plot
plt.savefig('Charts/LassoPrediction_with_Trendline.png')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load your dataset
data = pd.read_csv('data/LegoTrainingData.csv')  # Replace with the correct path to your CSV file

# Select relevant columns
data = data[['num_parts', 'num_figs', 'retail_price', 
             'num_unique_figs', 'set_rating', 'pop_price', 
             'return']]

# Split features and target
X = data.drop('return', axis=1)  # Replace 'target_column' with your target column name
y = data['return']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model (for regression)
xgb_regressor = xgb.XGBRegressor(
    n_estimators=100,  # Number of trees
    learning_rate=0.1,  # Step size shrinkage
    max_depth=6,       # Maximum tree depth
    subsample=0.8,     # Row sampling
    colsample_bytree=0.8,  # Column sampling
    random_state=42
)

# Fit the model
xgb_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Regression Results:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Classification: Ensure target is categorical
if not np.array_equal(y, y.astype(int)):
    print("Target variable is not categorical. Classification model cannot be used.")
else:
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb_classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred_class = xgb_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred_class)
    print("Classification Results:")
    print(f"Accuracy: {accuracy}")

# Optional: Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb.XGBRegressor(random_state=42),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=3,
                           verbose=2)

print("Starting Grid Search for Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the best model
best_model = xgb.XGBRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Predict and evaluate with the best model
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print("Best Model Results:")
print(f"Mean Squared Error: {mse_best}")
print(f"R2 Score: {r2_best}")
xgb_regressor.save_model('xgb_lego_model.json')