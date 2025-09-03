import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Load data
data = pd.read_excel("TestingCohort2.xlsx")

# 1. Handle missing values (represented by '?')
data = data.replace('?', np.nan)

# 2. Separate different types of variables
continuous_vars = ['Age', 'Body_Temperature', 'Systolic_Blood_Pressure', 
                  'Diastolic_Blood_Pressure', 'Heart_Rate', 'Blood_Oxygen_Saturation',
                  'Emergency Visits Last Year', 'Physician Age']

binary_vars = ['Gender', 'History: Hypertension', ' History: Diabetes', 
               'History: Heart Disease', 'History: COPD or Asthma', 'History: Stroke',
               'Activity Ability', 'CT', 'Ultrasound', 'ECG', 'MRI',
               'Initial Diagnosis: Respiratory System Disease',
               'Initial Diagnosis: Cardiovascular System Disease',
               'Initial Diagnosis: Digestive System Disease',
               'Initial Diagnosis: Nervous System Disease',
               'Initial Diagnosis: Uri?ry System Disease',
               'Initial Visit Date a Holiday']

ordinal_vars = ['Initial Triage Level', 'Physician Seniority', 'Initial Visit Schedule']

# 3. Handle continuous variables
continuous_imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

data[continuous_vars] = continuous_imputer.fit_transform(data[continuous_vars].astype(float))
data[continuous_vars] = scaler.fit_transform(data[continuous_vars])

# 4. Handle binary variables
binary_imputer = SimpleImputer(strategy='most_frequent')
label_encoder = LabelEncoder()

for col in binary_vars:
    # Convert Series to 2D array
    col_2d = data[col].values.reshape(-1, 1)
    # Perform missing value imputation
    imputed_values = binary_imputer.fit_transform(col_2d)
    # Convert to string and perform label encoding
    data[col] = label_encoder.fit_transform(imputed_values.ravel().astype(str))

# 5. Handle ordinal variables
ordinal_imputer = SimpleImputer(strategy='most_frequent')
ordinal_encoder = OrdinalEncoder()

# Combine ordinal variables into a 2D array
ordinal_data = data[ordinal_vars].values
# Perform missing value imputation
ordinal_imputed = ordinal_imputer.fit_transform(ordinal_data)
# Perform ordinal encoding
ordinal_encoded = ordinal_encoder.fit_transform(ordinal_imputed)

# Assign encoded values back to DataFrame
for i, col in enumerate(ordinal_vars):
    data[col] = ordinal_encoded[:, i]

# 6. Save preprocessed data
data.to_excel("StandardizedData.xlsx", index=False)

# Output simple statistics to verify preprocessing results
print("\nContinuous variables statistics:")
print(data[continuous_vars].describe())

print("\nBinary variables value statistics:")
for col in binary_vars:
    print(f"\n{col} unique values:", data[col].unique())

print("\nOrdinal variables value statistics:")
for col in ordinal_vars:
    print(f"\n{col} unique values:", data[col].unique())
