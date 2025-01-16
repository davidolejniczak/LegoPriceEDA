import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression, LassoCV

df = pd.read_csv('data/LegoStatsHistorical.csv')

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
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)
df_model = df_encoded[['set_id', 'num_parts', 'num_figs', 'retail_price', 
                       'num_unique_figs', 'set_rating', 'pop_price', 
                       'num_reviews', 'return']]
X = df_model.drop(['return'], axis=1)
y = df_model['return']


X = X.apply(pd.to_numeric, errors='coerce')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Training R2 Score:", r2_score(y_train, y_train_pred))
print("Training MAE Score:", mean_absolute_error(y_train, y_train_pred))
print("Test R2 Score:", r2_score(y_test, y_test_pred))
print("Test MAE Score:", mean_absolute_error(y_test, y_test_pred))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5, color='orange')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs Predicted (Test Set)")
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Charts/LSEPrediction.png')
plt.close()

'''Cross-Validation Regression'''
cross_val_r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("Cross-Validation R2 Score:", cross_val_r2_scores.mean())

mae = make_scorer(mean_absolute_error, greater_is_better=False)
cross_val_mae_scores = cross_val_score(model, X, y, cv=5, scoring=mae)
print("Cross-Validation MAE Score:", -cross_val_mae_scores.mean())

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
lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)

y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

print("Training R2 Score (Lasso):", r2_score(y_train, y_train_pred))
print("Training MAE Score (Lasso):", mean_absolute_error(y_train, y_train_pred))
print("Test R2 Score (Lasso):", r2_score(y_test, y_test_pred))
print("Test MAE Score (Lasso):", mean_absolute_error(y_test, y_test_pred))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5, color='green')
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
