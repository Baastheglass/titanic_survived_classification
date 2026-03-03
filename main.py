import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')

print("=" * 80)
print("TITANIC DATASET OVERVIEW")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset info:")
print(df.info())
print(f"\nBasic statistics:")
print(df.describe())
print(f"\nMissing values BEFORE cleaning:")
print(df.isnull().sum())

# ============================================================================
# HANDLE MISSING VALUES
# ============================================================================

print("\n" + "=" * 80)
print("HANDLING MISSING VALUES")
print("=" * 80)

# 1. Remove rows with missing Embarked values (only 2 records)
print(f"\nRemoving {df['Embarked'].isnull().sum()} rows with missing Embarked values...")
df = df.dropna(subset=['Embarked'])
print(f"✓ New dataset shape: {df.shape}")

# 2. Fill missing Age values using Random Forest Regression
print(f"\nFilling {df['Age'].isnull().sum()} missing Age values using Random Forest Regression...")

# Prepare features for Age prediction
df_age = df.copy()
df_age['Sex_numeric'] = df_age['Sex'].map({'male': 0, 'female': 1})
df_age['Embarked_numeric'] = df_age['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Features to use for Age prediction
age_features = ['Pclass', 'Sex_numeric', 'SibSp', 'Parch', 'Fare', 'Embarked_numeric']

# Split into rows with Age and rows without Age
df_with_age = df_age[df_age['Age'].notna()].copy()
df_without_age = df_age[df_age['Age'].isna()].copy()

# Train the regression model
X_train_age = df_with_age[age_features]
y_train_age = df_with_age['Age']

age_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
age_model.fit(X_train_age, y_train_age)

# Predict missing ages
X_predict_age = df_without_age[age_features]
predicted_ages = age_model.predict(X_predict_age)

# Fill missing ages in original dataframe
df.loc[df['Age'].isna(), 'Age'] = predicted_ages

print(f"✓ Filled {len(predicted_ages)} missing Age values")
print(f"  Mean predicted age: {predicted_ages.mean():.2f} years")
print(f"  Age range predicted: {predicted_ages.min():.2f} - {predicted_ages.max():.2f} years")

# 3. Fill missing Cabin values using Random Forest Classification
print(f"\nFilling {df['Cabin'].isnull().sum()} missing Cabin values using Random Forest Classification...")

# Extract cabin deck (first letter) from existing cabin values
df['Cabin_Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else None)

# Prepare features for Cabin prediction
df_cabin = df.copy()
df_cabin['Sex_numeric'] = df_cabin['Sex'].map({'male': 0, 'female': 1})
df_cabin['Embarked_numeric'] = df_cabin['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Features to use for Cabin prediction
cabin_features = ['Pclass', 'Sex_numeric', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_numeric']

# Split into rows with Cabin and rows without Cabin
df_with_cabin = df_cabin[df_cabin['Cabin_Deck'].notna()].copy()
df_without_cabin = df_cabin[df_cabin['Cabin_Deck'].isna()].copy()

# Train the classification model
X_train_cabin = df_with_cabin[cabin_features]
y_train_cabin = df_with_cabin['Cabin_Deck']

cabin_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
cabin_model.fit(X_train_cabin, y_train_cabin)

# Predict missing cabin decks
X_predict_cabin = df_without_cabin[cabin_features]
predicted_cabin_decks = cabin_model.predict(X_predict_cabin)

# Fill missing cabins in original dataframe (use just the deck letter)
df.loc[df['Cabin'].isna(), 'Cabin'] = predicted_cabin_decks
df.loc[df['Cabin_Deck'].isna(), 'Cabin_Deck'] = predicted_cabin_decks

print(f"✓ Filled {len(predicted_cabin_decks)} missing Cabin values")
print(f"  Predicted cabin deck distribution:")
unique, counts = np.unique(predicted_cabin_decks, return_counts=True)
for deck, count in zip(unique, counts):
    print(f"    Deck {deck}: {count} passengers")

# Model performance summary
print("\n" + "-" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("-" * 80)

# Age model - cross-validation on known ages
age_cv_scores = cross_val_score(age_model, X_train_age, y_train_age, 
                                 cv=5, scoring='neg_mean_absolute_error')
print(f"\nAge Prediction Model (Random Forest Regression):")
print(f"  Mean Absolute Error (CV): {-age_cv_scores.mean():.2f} years")
print(f"  Feature Importances:")
for feature, importance in sorted(zip(age_features, age_model.feature_importances_), 
                                  key=lambda x: x[1], reverse=True):
    print(f"    {feature:20s}: {importance:.4f}")

# Cabin model - accuracy on known cabins
cabin_cv_scores = cross_val_score(cabin_model, X_train_cabin, y_train_cabin, 
                                   cv=5, scoring='accuracy')
print(f"\nCabin Prediction Model (Random Forest Classification):")
print(f"  Accuracy (CV): {cabin_cv_scores.mean():.2%}")
print(f"  Feature Importances:")
for feature, importance in sorted(zip(cabin_features, cabin_model.feature_importances_), 
                                  key=lambda x: x[1], reverse=True):
    print(f"    {feature:20s}: {importance:.4f}")

print("\n" + "=" * 80)
print(f"Missing values AFTER cleaning:")
print(df.isnull().sum())
print("=" * 80)

# ============================================================================
# FEATURE ENGINEERING - Create FamilySize
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Create FamilySize feature (combines SibSp + Parch + 1 for the passenger)
print("\nCreating FamilySize feature...")
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

print(f"✓ FamilySize feature created")
print(f"  Family size distribution:")
family_size_dist = df['FamilySize'].value_counts().sort_index()
for size, count in family_size_dist.items():
    print(f"    Size {size}: {count} passengers ({count/len(df)*100:.1f}%)")

print(f"\n  Average family size: {df['FamilySize'].mean():.2f}")
print(f"  Family size range: {df['FamilySize'].min()} - {df['FamilySize'].max()}")

# Analyze survival by family size
print("\n  Survival rate by family size:")
for size in sorted(df['FamilySize'].unique()):
    survival_rate = df[df['FamilySize'] == size]['Survived'].mean() * 100
    count = len(df[df['FamilySize'] == size])
    print(f"    Size {size}: {survival_rate:.1f}% (n={count})")

print("\n" + "=" * 80)

# Create visualizations
fig = plt.figure(figsize=(16, 12))

# 1. Survival Rate
plt.subplot(2, 3, 1)
survival_counts = df['Survived'].value_counts()
plt.bar(['Did not survive', 'Survived'], survival_counts.values, color=['#ff6b6b', '#51cf66'])
plt.title('Survival Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Count')
for i, v in enumerate(survival_counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')

# 2. Survival by Gender
plt.subplot(2, 3, 2)
survival_by_sex = pd.crosstab(df['Sex'], df['Survived'], normalize='index') * 100
survival_by_sex.plot(kind='bar', ax=plt.gca(), color=['#ff6b6b', '#51cf66'])
plt.title('Survival Rate by Gender', fontsize=14, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(['Did not survive', 'Survived'])

# 3. Survival by Passenger Class
plt.subplot(2, 3, 3)
survival_by_class = pd.crosstab(df['Pclass'], df['Survived'], normalize='index') * 100
survival_by_class.plot(kind='bar', ax=plt.gca(), color=['#ff6b6b', '#51cf66'])
plt.title('Survival Rate by Passenger Class', fontsize=14, fontweight='bold')
plt.xlabel('Passenger Class')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(['Did not survive', 'Survived'])

# 4. Age Distribution
plt.subplot(2, 3, 4)
df['Age'].hist(bins=30, edgecolor='black', alpha=0.7)
plt.title('Age Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')

# 5. Fare Distribution
plt.subplot(2, 3, 5)
df['Fare'].hist(bins=50, edgecolor='black', alpha=0.7, color='#339af0')
plt.title('Fare Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Fare')
plt.ylabel('Frequency')

# 6. Survival by Embarked Port
plt.subplot(2, 3, 6)
survival_by_embarked = pd.crosstab(df['Embarked'], df['Survived'], normalize='index') * 100
survival_by_embarked.plot(kind='bar', ax=plt.gca(), color=['#ff6b6b', '#51cf66'])
plt.title('Survival Rate by Embarked Port', fontsize=14, fontweight='bold')
plt.xlabel('Embarked Port')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(['Did not survive', 'Survived'])

plt.tight_layout()
plt.savefig('titanic_visualizations.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualizations to 'titanic_visualizations.png'")
plt.show()

# ============================================================================
# CORRELATION MATRIX - Features that correlate with Survival
# ============================================================================

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS WITH SURVIVAL")
print("=" * 80)

# Prepare data for correlation analysis
# Convert categorical variables to numerical
df_corr = df.copy()

# Convert Sex to numerical (0 = male, 1 = female)
df_corr['Sex_numeric'] = df_corr['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked to numerical
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
df_corr['Embarked_numeric'] = df_corr['Embarked'].map(embarked_mapping)

# Select numerical features for correlation (using FamilySize instead of SibSp and Parch)
numerical_features = ['Survived', 'Pclass', 'Sex_numeric', 'Age', 'FamilySize', 
                      'Fare', 'Embarked_numeric']

# Create correlation matrix
correlation_data = df_corr[numerical_features].corr()

# Get correlations with 'Survived' and sort them
survival_correlations = correlation_data['Survived'].sort_values(ascending=False)
print("\nCorrelation of features with Survival (sorted):")
print("-" * 50)
for feature, corr in survival_correlations.items():
    if feature != 'Survived':
        print(f"{feature:20s}: {corr:>7.4f}")

# Create correlation matrix heatmap
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Full correlation matrix
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
            fmt='.3f', square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=axes[0])
axes[0].set_title('Correlation Matrix - All Features', fontsize=16, fontweight='bold', pad=20)

# Bar plot of correlations with Survival
survival_corr_sorted = survival_correlations.drop('Survived')
colors = ['#51cf66' if x > 0 else '#ff6b6b' for x in survival_corr_sorted.values]
axes[1].barh(range(len(survival_corr_sorted)), survival_corr_sorted.values, color=colors)
axes[1].set_yticks(range(len(survival_corr_sorted)))
axes[1].set_yticklabels(survival_corr_sorted.index)
axes[1].set_xlabel('Correlation Coefficient', fontsize=12)
axes[1].set_title('Feature Correlation with Survival', fontsize=16, fontweight='bold', pad=20)
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1].grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(survival_corr_sorted.values):
    axes[1].text(v + 0.01 if v > 0 else v - 0.01, i, f'{v:.3f}', 
                va='center', ha='left' if v > 0 else 'right', fontweight='bold')

plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved correlation matrix to 'correlation_matrix.png'")
plt.show()

# Additional insights
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("\nTop 3 features positively correlated with survival:")
positive_corr = survival_correlations[survival_correlations > 0].drop('Survived').head(3)
for i, (feature, corr) in enumerate(positive_corr.items(), 1):
    print(f"{i}. {feature}: {corr:.4f}")

print("\nTop 3 features negatively correlated with survival:")
negative_corr = survival_correlations[survival_correlations < 0].head(3)
for i, (feature, corr) in enumerate(negative_corr.items(), 1):
    print(f"{i}. {feature}: {corr:.4f}")

print("\nSurvival statistics:")
print(f"Overall survival rate: {df['Survived'].mean() * 100:.2f}%")
print(f"Female survival rate: {df[df['Sex'] == 'female']['Survived'].mean() * 100:.2f}%")
print(f"Male survival rate: {df[df['Sex'] == 'male']['Survived'].mean() * 100:.2f}%")
print(f"1st class survival rate: {df[df['Pclass'] == 1]['Survived'].mean() * 100:.2f}%")
print(f"2nd class survival rate: {df[df['Pclass'] == 2]['Survived'].mean() * 100:.2f}%")
print(f"3rd class survival rate: {df[df['Pclass'] == 3]['Survived'].mean() * 100:.2f}%")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# ============================================================================
# SURVIVAL PREDICTION - SIMPLE CLASSIFIER
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING SURVIVAL PREDICTION CLASSIFIER")
print("=" * 80)

# Prepare features for classification
print("\nPreparing features for classification...")

# Select relevant features (using FamilySize instead of SibSp and Parch)
feature_columns = ['Pclass', 'Sex_numeric', 'Age', 'FamilySize', 'Fare', 'Embarked_numeric']

X = df_corr[feature_columns]
y = df_corr['Survived']

print(f"✓ Selected {len(feature_columns)} features:")
for feat in feature_columns:
    print(f"  - {feat}")

# Split data: 75% training, 25% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n✓ Data split complete:")
print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df_corr)*100:.1f}%)")
print(f"  Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df_corr)*100:.1f}%)")
print(f"  Training survival rate: {y_train.mean()*100:.2f}%")
print(f"  Testing survival rate: {y_test.mean()*100:.2f}%")

# Train Random Forest Classifier
print("\n" + "-" * 80)
print("Training Random Forest Classifier...")
print("-" * 80)

classifier = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    min_samples_split=10,
    random_state=42
)
classifier.fit(X_train, y_train)

print("✓ Model training complete!")

# Make predictions
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)
y_test_proba = classifier.predict_proba(X_test)[:, 1]

# Evaluate model
print("\n" + "-" * 80)
print("MODEL EVALUATION")
print("-" * 80)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"\nAccuracy:")
print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  ROC-AUC Score:     {roc_auc:.4f}")

# Classification report
print(f"\nDetailed Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred, 
                          target_names=['Did not survive', 'Survived']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"Confusion Matrix:")
print(f"                Predicted")
print(f"                Not Survived  Survived")
print(f"Actual Not     [{cm[0,0]:^12d}  {cm[0,1]:^8d}]")
print(f"Survived       [{cm[1,0]:^12d}  {cm[1,1]:^8d}]")

# Feature Importances
print(f"\n" + "-" * 80)
print("FEATURE IMPORTANCE")
print("-" * 80)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature importances (sorted):")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.4f}")

# Visualizations
fig = plt.figure(figsize=(16, 10))

# 1. Feature Importances
plt.subplot(2, 3, 1)
plt.barh(range(len(feature_importance)), feature_importance['importance'], color='#4CAF50')
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Importance', fontweight='bold')
plt.title('Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# 2. Confusion Matrix Heatmap
plt.subplot(2, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Did not survive', 'Survived'],
            yticklabels=['Did not survive', 'Survived'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.xlabel('Predicted', fontweight='bold')

# 3. ROC Curve
plt.subplot(2, 3, 3)
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})', color='#2196F3')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)

# 4. Prediction Probability Distribution
plt.subplot(2, 3, 4)
survived_proba = y_test_proba[y_test == 1]
died_proba = y_test_proba[y_test == 0]
plt.hist(died_proba, bins=20, alpha=0.6, label='Did not survive', color='#ff6b6b', edgecolor='black')
plt.hist(survived_proba, bins=20, alpha=0.6, label='Survived', color='#51cf66', edgecolor='black')
plt.xlabel('Predicted Probability of Survival', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 5. Training vs Testing Accuracy
plt.subplot(2, 3, 5)
accuracies = [train_accuracy, test_accuracy]
labels = ['Training', 'Testing']
colors = ['#4CAF50', '#2196F3']
bars = plt.bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black')
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Training vs Testing Accuracy', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{acc:.2%}', ha='center', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 6. Prediction Examples
plt.subplot(2, 3, 6)
# Show actual vs predicted for first 50 test samples
sample_size = min(50, len(y_test))
sample_indices = range(sample_size)
plt.scatter(sample_indices, y_test.iloc[:sample_size], alpha=0.6, s=100, 
           label='Actual', marker='o', color='#2196F3')
plt.scatter(sample_indices, y_test_pred[:sample_size], alpha=0.6, s=50, 
           label='Predicted', marker='x', color='#FF5722', linewidths=2)
plt.xlabel('Sample Index', fontweight='bold')
plt.ylabel('Survival (0=No, 1=Yes)', fontweight='bold')
plt.title(f'Actual vs Predicted (First {sample_size} Test Samples)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.yticks([0, 1], ['Did not survive', 'Survived'])

plt.tight_layout()
plt.savefig('survival_classifier_results.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved classifier results to 'survival_classifier_results.png'")
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SURVIVAL PREDICTION - FINAL SUMMARY")
print("=" * 80)

correct_predictions = (y_test_pred == y_test).sum()
total_predictions = len(y_test)

print(f"\n✓ Model successfully trained on {len(X_train)} samples")
print(f"✓ Model tested on {len(X_test)} samples")
print(f"\n📊 Performance Metrics:")
print(f"  • Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  • Correct Predictions: {correct_predictions}/{total_predictions}")
print(f"  • ROC-AUC Score: {roc_auc:.4f}")
print(f"\n🎯 Most Important Features:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")

print("\n💡 Interpretation:")
if test_accuracy > 0.8:
    print(f"  The classifier achieves excellent accuracy ({test_accuracy*100:.1f}%) on unseen data!")
elif test_accuracy > 0.75:
    print(f"  The classifier achieves good accuracy ({test_accuracy*100:.1f}%) on unseen data.")
else:
    print(f"  The classifier achieves moderate accuracy ({test_accuracy*100:.1f}%) on unseen data.")

print(f"  The model is {'slightly overfitting' if train_accuracy - test_accuracy > 0.05 else 'well-balanced'}")
print(f"  based on the difference between training ({train_accuracy*100:.1f}%) and test ({test_accuracy*100:.1f}%) accuracy.")

print("\n" + "=" * 80)
print("ALL ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. titanic_visualizations.png - Dataset visualizations")
print("  2. correlation_matrix.png - Correlation analysis")
print("  3. survival_classifier_results.png - Classifier performance")
print("=" * 80)
