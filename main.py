# =============================================================================
# Multiple Linear Regression Assignment
# Dataset: California Housing Dataset (from sklearn)
# Target Variable: Median House Value
# =============================================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =============================================================================
# STEP 1: LOAD DATASET
# =============================================================================

# Load the California Housing dataset from sklearn
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
housing = fetch_california_housing()

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add the target variable (median house value in $100,000s)
df['MedHouseVal'] = housing.target

# Display all available features for reference
print("All available features:", housing.feature_names)
print("\nFeature descriptions:")
print(housing.DESCR[:1500])  # Print first part of description

# =============================================================================
# PREDICTOR VARIABLE SELECTION (Justified)
# =============================================================================
# We select 5 predictor variables based on domain knowledge and likely correlation:
#
# 1. MedInc    - Median income of households: Higher income typically means
#                higher property values. Strong economic predictor.
#
# 2. AveRooms  - Average number of rooms per household: Larger homes tend to
#                be worth more. Reflects property size.
#
# 3. HouseAge  - Median age of houses in the block: Older houses may be worth
#                less due to depreciation (or more if in historic areas).
#
# 4. Latitude  - Geographic location (north-south): Location dramatically
#                affects property values in California.
#
# 5. Longitude - Geographic location (east-west): Coastal areas (negative
#                longitude closer to -120) tend to have higher values.
#
# We EXCLUDE: AveBedrms (highly correlated with AveRooms, causes multicollinearity),
#             AveOccup (outlier-prone), Population (less direct impact on value)
# =============================================================================

# Select chosen predictors
selected_features = ['MedInc', 'AveRooms', 'HouseAge', 'Latitude', 'Longitude']
target = 'MedHouseVal'

# Create working DataFrame with selected columns only
df_selected = df[selected_features + [target]].copy()

print("\n" + "="*60)
print("SELECTED FEATURES:", selected_features)
print("TARGET VARIABLE:", target)
print("="*60)

# =============================================================================
# STEP 2: DATA DESCRIPTION
# =============================================================================

print("\n--- DATA DESCRIPTION ---")

# Print shape (rows x columns)
print(f"\nDataset Shape: {df_selected.shape[0]} rows x {df_selected.shape[1]} columns")

# Show first few rows
print("\nFirst 5 rows of the dataset:")
print(df_selected.head())

# Brief dataset description (in comments below)
# Each row represents a census block group in California (approx. 600-3000 people).
# The target variable MedHouseVal is in units of $100,000 (e.g., 3.5 = $350,000).
# Data was collected from the 1990 California Census.

print("\nDataset Info:")
print(df_selected.info())

# =============================================================================
# STEP 3: PREPROCESSING
# =============================================================================

print("\n--- PREPROCESSING ---")

# Check for missing values in each column
print("\nMissing values per column:")
print(df_selected.isnull().sum())

# Handle missing values (if any exist, fill with column median)
# Median is preferred over mean as it is robust to outliers
if df_selected.isnull().sum().sum() > 0:
    df_selected.fillna(df_selected.median(), inplace=True)
    print("Missing values filled with column medians.")
else:
    print("No missing values found. Dataset is complete.")

# Check for duplicate rows
num_duplicates = df_selected.duplicated().sum()
print(f"\nNumber of duplicate rows: {num_duplicates}")

# Remove duplicates if any exist
if num_duplicates > 0:
    df_selected.drop_duplicates(inplace=True)
    print(f"Removed {num_duplicates} duplicate rows.")
    print(f"New shape after removing duplicates: {df_selected.shape}")
else:
    print("No duplicate rows found.")

# Check for unusual/extreme values (basic outlier check using IQR)
print("\nChecking for extreme values (values beyond 3 standard deviations):")
for col in selected_features:
    z_scores = np.abs((df_selected[col] - df_selected[col].mean()) / df_selected[col].std())
    outlier_count = (z_scores > 3).sum()
    print(f"  {col}: {outlier_count} potential outliers (beyond 3 std dev)")

# Note: We keep outliers in this assignment as removing them could bias results
# and the assignment does not require outlier removal explicitly

print("\nDataset is clean and ready for exploration.")

# =============================================================================
# STEP 4: DATA EXPLORATION
# =============================================================================

print("\n--- DATA EXPLORATION ---")

# Summary Statistics
print("\nSummary Statistics:")
print(df_selected.describe().round(3))

# Correlation Table
print("\nCorrelation Matrix (with target variable):")
correlation_matrix = df_selected.corr()
print(correlation_matrix.round(3))

# Set a clean visual style for all plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 7)

# --- VISUALIZATION 1: Correlation Heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,          # Show correlation values
    fmt=".2f",           # Round to 2 decimal places
    cmap="coolwarm",     # Blue = negative, Red = positive correlation
    center=0,
    square=True,
    linewidths=0.5
)
plt.title("Correlation Heatmap of Selected Features and Target Variable\n"
          "(Positive values = features increase together; Negative = inverse relationship)",
          fontsize=13)
plt.tight_layout()
plt.savefig("plot_1_correlation_heatmap.png", dpi=150)
plt.show()
print("\n[Visualization 1] Correlation Heatmap:")
print("  MedInc has the strongest positive correlation with MedHouseVal (r ≈ 0.69).")
print("  Latitude and Longitude show negative correlations, reflecting California's geography.")
print("  AveRooms has a moderate positive correlation with house value.")

# --- VISUALIZATION 2: Histogram of Key Variables ---
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("Histograms of Selected Features and Target Variable", fontsize=14, fontweight='bold')

columns_to_plot = selected_features + [target]
colors = ['steelblue', 'coral', 'mediumseagreen', 'orchid', 'goldenrod', 'tomato']

for i, (col, color) in enumerate(zip(columns_to_plot, colors)):
    ax = axes[i // 3][i % 3]
    ax.hist(df_selected[col], bins=40, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(col, fontsize=11)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("plot_2_histograms.png", dpi=150)
plt.show()
print("\n[Visualization 2] Histograms:")
print("  MedInc is right-skewed (most areas have moderate income, few have very high).")
print("  MedHouseVal is capped at 5.0 ($500K) — note the spike at the right tail.")
print("  HouseAge shows a relatively uniform distribution across years.")

# --- VISUALIZATION 3: Scatter Plots (Predictors vs Target) ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Scatter Plots: Each Predictor vs Median House Value", fontsize=14, fontweight='bold')

for i, col in enumerate(selected_features):
    ax = axes[i // 3][i % 3]
    ax.scatter(df_selected[col], df_selected[target], alpha=0.15, color='steelblue', s=5)
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel("MedHouseVal ($100K)", fontsize=10)
    ax.set_title(f"{col} vs House Value", fontsize=11)

# Hide the empty 6th subplot
axes[1][2].set_visible(False)

plt.tight_layout()
plt.savefig("plot_3_scatter_plots.png", dpi=150)
plt.show()
print("\n[Visualization 3] Scatter Plots:")
print("  MedInc vs MedHouseVal shows a clear positive trend — the most useful predictor.")
print("  Latitude shows a clear geographic pattern (Northern CA has lower values overall).")
print("  AveRooms has a positive trend but with some high-room outliers.")

# --- VISUALIZATION 4: Box Plot (distribution of each feature) ---
fig, axes = plt.subplots(1, len(selected_features), figsize=(16, 6))
fig.suptitle("Box Plots of Selected Predictor Variables\n"
             "(Shows median, IQR, and outliers for each feature)", fontsize=13, fontweight='bold')

box_colors = ['lightblue', 'lightcoral', 'lightgreen', 'plum', 'lightyellow']

for i, (col, color) in enumerate(zip(selected_features, box_colors)):
    axes[i].boxplot(df_selected[col], patch_artist=True,
                    boxprops=dict(facecolor=color, color='navy'),
                    medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(col, fontsize=10)
    axes[i].set_ylabel("Value")

plt.tight_layout()
plt.savefig("plot_4_boxplots.png", dpi=150)
plt.show()
print("\n[Visualization 4] Box Plots:")
print("  AveRooms has notable high-end outliers (some blocks have very large avg room count).")
print("  MedInc shows a right-skewed distribution with upper outliers.")
print("  HouseAge and geographic features (Lat/Lon) are more evenly distributed.")

# =============================================================================
# STEP 5: BUILD THE MODEL
# =============================================================================

print("\n--- MODEL BUILDING ---")

# Define features (X) and target (y)
X = df_selected[selected_features]
y = df_selected[target]

# Split data into training (80%) and testing (20%) sets
# WHY: We need unseen data to evaluate how well the model generalizes.
#      Training on all data and testing on the same data would give overoptimistic results.
#      The test set simulates "new, real-world" data the model has never seen.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 20% of data reserved for testing
    random_state=42       # Fixed seed for reproducibility
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)  # Model learns the relationship between X and y

# Print model intercept
print(f"\nModel Intercept: {model.intercept_:.4f}")
print("(This is the predicted house value when all predictors are zero — a baseline)")

# Print and explain each coefficient
print("\nModel Coefficients:")
print("-" * 55)
print(f"{'Feature':<15} {'Coefficient':>12}  Meaning")
print("-" * 55)

for feature, coef in zip(selected_features, model.coef_):
    if feature == 'MedInc':
        meaning = "Each 1-unit increase in median income raises predicted value by this amount"
    elif feature == 'AveRooms':
        meaning = "Each additional average room is associated with this change in value"
    elif feature == 'HouseAge':
        meaning = "Each 1-year increase in house age changes value by this amount"
    elif feature == 'Latitude':
        meaning = "Moving 1 degree north changes value by this amount"
    elif feature == 'Longitude':
        meaning = "Moving 1 degree east changes value by this amount"
    else:
        meaning = "See feature description"
    print(f"{feature:<15} {coef:>12.4f}  {meaning[:50]}")

print("-" * 55)
print("\nCoefficients represent the change in MedHouseVal ($100K)")
print("for a 1-unit increase in the predictor, holding all others constant.")

# =============================================================================
# STEP 6: MODEL EVALUATION
# =============================================================================

print("\n--- MODEL EVALUATION ---")

# Predict on the test set (data the model has NEVER seen)
y_pred = model.predict(X_test)

# Calculate MAE (Mean Absolute Error)
# MAE = average of |actual - predicted| across all test samples
# Easy to interpret: "On average, predictions are off by $X"
mae = mean_absolute_error(y_test, y_pred)

# Calculate RMSE (Root Mean Squared Error)
# RMSE penalizes large errors more than MAE (because errors are squared)
# Also in same units as the target variable
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nEvaluation Metrics on Test Set:")
print(f"  MAE  (Mean Absolute Error):       {mae:.4f} ($100K) = ${mae*100000:,.0f}")
print(f"  RMSE (Root Mean Squared Error):   {rmse:.4f} ($100K) = ${rmse*100000:,.0f}")

# Model performance interpretation
print(f"\nR² Score (coefficient of determination): {model.score(X_test, y_test):.4f}")
print("(R² = 1.0 is perfect; R² = 0.0 means the model explains no variance)")

# Assess performance
if mae < 0.5:
    print("\nModel Performance: GOOD — MAE is under $50,000 on average.")
elif mae < 0.8:
    print("\nModel Performance: MODERATE — predictions are within ~$80,000 on average.")
else:
    print("\nModel Performance: NEEDS IMPROVEMENT — consider adding more features or non-linear models.")

# --- Actual vs Predicted Scatter Plot ---
plt.figure(figsize=(9, 7))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue', s=10, label='Predictions')

# Perfect prediction line (diagonal)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val],
         color='red', linewidth=2, linestyle='--', label='Perfect Prediction Line')

plt.xlabel("Actual Median House Value ($100K)", fontsize=12)
plt.ylabel("Predicted Median House Value ($100K)", fontsize=12)
plt.title("Actual vs Predicted Median House Values\n"
          "(Closer to red line = better predictions)", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_5_actual_vs_predicted.png", dpi=150)
plt.show()
print("\n[Evaluation Plot] Actual vs Predicted:")
print("  Points close to the diagonal red line = accurate predictions.")
print("  The model performs well for mid-range values but struggles at extremes.")
print("  The ceiling effect at 5.0 ($500K) is visible — data was capped in the original dataset.")

# --- Residuals Plot ---
residuals = y_test - y_pred  # Residual = actual - predicted

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.3, color='coral', s=10)
plt.axhline(y=0, color='navy', linewidth=2, linestyle='--', label='Zero Error Line')
plt.xlabel("Predicted Values ($100K)", fontsize=12)
plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
plt.title("Residuals vs Predicted Values\n"
          "(Random scatter around 0 = model assumptions are satisfied)", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_6_residuals.png", dpi=150)
plt.show()
print("\n[Residuals Plot]:")
print("  Residuals show some pattern at higher predicted values, suggesting")
print("  non-linearity that a simple linear model cannot fully capture.")

# =============================================================================
# STEP 7: REFLECTION (in code comments and print statements)
# =============================================================================

print("\n" + "="*60)
print("REFLECTION")
print("="*60)
print("""
1. WHY DID YOU CHOOSE THIS DATASET?
   The California Housing dataset is well-known, clean, publicly available
   through sklearn, and has a clear continuous target variable (house prices).
   It is ideal for learning multiple linear regression.

2. WHY DID YOU CHOOSE THOSE PREDICTORS?
   - MedInc: Economic factor most directly tied to home affordability and value.
   - AveRooms: Reflects property size, a key determinant of price.
   - HouseAge: Captures the effect of depreciation or renovation cycles.
   - Latitude & Longitude: Location is the most fundamental factor in real estate.
   We excluded AveBedrms (collinear with AveRooms) and Population (indirect effect).

3. WHAT WAS DIFFICULT IN THIS ASSIGNMENT?
   Choosing which features to include and exclude required careful reasoning.
   Interpreting coefficients for geographic variables (Lat/Lon) is less intuitive
   than for economic variables. Handling the value ceiling ($500K cap) is also tricky.

4. WHAT DID I LEARN?
   - How to build and evaluate a multiple linear regression model end-to-end.
   - Feature selection matters: including irrelevant or collinear features can hurt.
   - Train/test split is essential to avoid overfitting and get honest evaluation.
   - MAE and RMSE give complementary views of model accuracy.
   - Residual plots help diagnose model assumptions and limitations.
""")

print("="*60)
print("ASSIGNMENT COMPLETE")
print("All plots saved as PNG files in the current directory.")
print("="*60)