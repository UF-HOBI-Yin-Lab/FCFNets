import pandas as pd
import numpy as np
import ast 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

###Step 1: Manully removed data###
data_df = pd.read_csv('./dataset/MPData/hf_manual_data.csv')
###set value 'TNP' as Null##
data_df = data_df.replace('TNP', np.nan)
# ####set int to float###
# data_df['ALBUMIN_result'] = data_df['ALBUMIN_result'].astype(float)
# data_df['GLUCOSE_result'] = data_df['GLUCOSE_result'].astype(float)

# Function to clean and convert the column
def preprocess_spe_values(value):
    try:
        # Return NaN for null or None values
        if pd.isna(value):
            return np.nan
        # Ensure value is treated as a string for substring checks
        value_str = str(value)
        # Return original value for special cases with '>' or '<'
        if '>' in value_str or '<' in value_str:
            return value
        # Convert valid numeric values to float
        return float(value)
    except ValueError:
        # Return original value for unexpected errors
        return value
    
###Step 2: Remove missing rate >30% features, add the label column, and save it into file### 
###try this###
data_mis_df = data_df.loc[:, data_df.isnull().mean() < 0.3]
pos_info_df = pd.read_csv('./dataset/MPData/pos.csv', low_memory=False)
neg_info_df = pd.read_csv('./dataset/MPData/neg.csv', low_memory=False)
pos_ids, neg_ids = pos_info_df['MRN_GNV'].tolist(), neg_info_df['MRN_GNV'].tolist()
data_mis_df['label'] = data_mis_df['MRN_GNV'].apply(lambda x: 1 if x in pos_ids else (0 if x in neg_ids else None))
data_mis_df.to_csv('./dataset/MPData/tem/hf_manual_misrate_data_v1.csv', index=False)

###Step 3: Modify inconsistent values in some features with values ['2+', '>=500']###
def process_ope_value(value, scale):
    try:
        if isinstance(value, str):
            value = value.strip()  # Remove leading/trailing spaces
            if '>=' in value or '<=' in value or '=' in value:
                base_value = float(value.replace('>=', '').replace('<=', '').replace('=', '').strip())
                return base_value
            elif '>' in value or '+' in value:
                base_value = float(value.replace('>', '').replace('+', '').strip())
                return base_value + scale
            elif '<' in value or '-' in value:
                base_value = float(value.replace('<', '').replace('-', '').strip())
                return base_value - scale
            else:
                # If it is a plain string without numeric conversions, return it as is
                return value
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return value  # Non-numeric and non-string, return as is
    except Exception as e:
        print(f"Error processing value '{value}': {e}")
        return None  # Return None for any exceptions

# Function to classify mixed values
def process_mix_value(value, rules):
    try:
        # Check numeric ranges
        numeric_value = float(value)
        for (low, high), label in rules.items():
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                if low < numeric_value <= high:
                    return label
    except ValueError:
        # Handle specific non-numeric values
        return value
    return value


#lower all strings in df
data_mis_df = data_mis_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
#deal with operation values
operation_cols = [('EGFR_result', 1), ('BILIRUBIN TOTAL_result', 0.1), ('ALKALINE PHOSPHATASE_result', 1), ('PLATELET COUNT_result', 1), ('CO2_result', 1),  \
                     ('ANION GAP_result', 1), ('HEMOGLOBIN A1C_result', 0.1), ('BILIRUBIN DIRECT_result', 0.1), ('GLUCOSE (METER)_result', 1), \
                  ('GLUCOSE UA_result', 1), ('KETONES UA_result',1), ('PROTEIN UA_result', 1), ('UROBILINOGEN UA_result', 0.1)]
for (col, ope) in operation_cols:
    data_mis_df[col] = data_mis_df[col].apply(lambda x: process_ope_value(x, ope))
    
#deal with the mixed values
mixed_cols = [
    ('GLUCOSE UA_result', {(0, 50): 'trace', (50, 150): 'small', (150, 500): 'moderate', (500, 5000): 'large'}),
    ('KETONES UA_result', {(0, 15): 'trace', (15, 40): 'small', (40, 80): 'moderate', (80, 200): 'large', 'positive': 'trace'}),
    ('PROTEIN UA_result', {(0, 30): 'trace', (30, 100): 'small', (100, 300): 'moderate', (300, 1000): 'large'}),
    ('UROBILINOGEN UA_result', {(0, 1.0): 'normal', (1.0, 2.0): 'small', (2.0, 4.0): 'moderate', (4.0, 10.0): 'large'})
]
# Apply the rules to each column
for col, rules in mixed_cols:
    data_mis_df[col] = data_mis_df[col].apply(lambda x: process_mix_value(x, rules))
data_mis_df.to_csv('./dataset/MPData/tem/hf_manual_misrate_data_v2.csv', index=False)

###Step 4: Min-max normalization###
# Function to convert mixed types to float
def convert_mixed_to_float(df, col):
    # Check if the column contains numeric values
    if df[col].apply(lambda x: isinstance(x, (int, float)) or pd.api.types.is_numeric_dtype(x)).all():
        # Convert the column to float
        return df[col].astype(float)
    return df[col]  # Leave column as-is if not numeric

#first transform the value with mixed types, i.e., float and int, into float
for col in data_mis_df.columns:
    try:
        data_mis_df[col] = convert_mixed_to_float(data_mis_df, col)
    except Exception as e:
        print(f"Error converting column '{col}': {e}")
        
#then start to normalize the continuing vlaues
scaler = MinMaxScaler()
numerical_features = data_mis_df.select_dtypes(include=np.number)
# print(numerical_features.head())
norm_fea_cols = [fea for fea in numerical_features.columns if fea != 'MRN_GNV']
# print(norm_fea_cols)
df_norm = pd.DataFrame(data_mis_df, columns=norm_fea_cols)
data_mis_df[df_norm.columns] = scaler.fit_transform(df_norm)
data_mis_df.to_csv('./dataset/MPData/tem/hf_manual_misrate_data_v3.csv', index=False)

###Step 5:Encode categorical features with one-hot###
# Step 1: Identify categorical columns
categorical_cols = data_mis_df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols_bp = categorical_cols.copy()
categorical_cols = [col for col in categorical_cols if col not in ["ICD10", "prescription_history_codes"]]
# Step 2: Replace NaN with a placeholder only in categorical columns
df_temp = data_mis_df.copy()
df_temp[categorical_cols] = df_temp[categorical_cols].fillna('NaN')
# Step 3: One-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_df = pd.DataFrame(
    encoder.fit_transform(df_temp[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols)
)
# # Step 4: Remove '_NaN' columns
encoded_df = encoded_df[[col for col in encoded_df.columns if not col.endswith('_NaN')]]
# encoded_df.to_csv('./dataset/MPData/tem/hf_manual_misrate_data_tt1.csv', index=False)

if "ICD10" in data_mis_df.columns:
    data_mis_df['ICD10'] = data_mis_df['ICD10'].apply(ast.literal_eval)
    mlb = MultiLabelBinarizer()
    one_hot_encoded = mlb.fit_transform(data_mis_df['ICD10'])
    encoded_icd_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_)
    # print('encoded_icd_df', encoded_icd_df.columns)
# # encoded_icd_df.to_csv('./dataset/MPData/tem/hf_manual_misrate_data_icd.csv', index=False)

if "prescription_history_codes" in data_mis_df.columns:
    # Function to safely evaluate or handle malformed entries
    def safe_literal_eval(val):
        try:
            if isinstance(val, str):
                # Attempt to parse the string into a Python list
                return ast.literal_eval(val)
            elif pd.isna(val):  # Handle NaN
                return []
            else:  # Already a list or unexpected type
                return val
        except (ValueError, SyntaxError):
            # Handle malformed cases
            return []
    data_mis_df['prescription_history_codes'] = data_mis_df['prescription_history_codes'].apply(safe_literal_eval)
    data_mis_df['prescription_history_codes'] = data_mis_df['prescription_history_codes'].apply(lambda codes: [f'pre_{str(code)}' for code in codes])
    mlb = MultiLabelBinarizer()
    binarized = mlb.fit_transform(data_mis_df['prescription_history_codes'])
    encoded_pres_df = pd.DataFrame(binarized, columns=mlb.classes_)
#     print('Total number of medication codes are', len(encoded_pres_df))
# encoded_pres_df.to_csv('./dataset/MPData/tem/hf_manual_misrate_data_pre.csv', index=False)
data_mis_df = pd.concat([data_mis_df.drop(columns=categorical_cols_bp), encoded_df, encoded_icd_df, encoded_pres_df], axis=1)
##replace icd code with name
icd10 = pd.read_csv("./dataset/MPData/ICD10_labels.tsv", sep='\t')
rename_dict = dict(zip(icd10['ICD10_code'].str.lower(), icd10['ICD10_label']))  
data_mis_df.rename(columns=rename_dict, inplace=True) 
data_mis_df['label'] = lbl
data_mis_df.to_csv('./dataset/MPData/tem/hf_manual_misrate_data_v4n.csv', index=False)

###Step 6: Feature selection (3-4 techniques)###
##Step 6.1: Remove zero-variance features##
###try this###
var_thres = 0.1
no_touch_cols = ['MRN_GNV', 'label']

var_thresh = VarianceThreshold(threshold=var_thres)
filtered_data = var_thresh.fit_transform(data_mis_df.drop(columns=no_touch_cols))
filtered_columns = data_mis_df.drop(columns=no_touch_cols).columns[var_thresh.get_support()]
# Filter dataset
df_var = pd.DataFrame(filtered_data, columns=filtered_columns, index=data_mis_df.index)
print(f"Selected features based on zero-variance (< {var_thres}): {len(filtered_columns)}, {len(df_var.columns)}")
df_var['label'] = data_mis_df['label']
df_var.to_csv('./dataset/MPData/tem/hf_manual_misrate_data_v5n.csv', index=False)

##Step 6.2: Pearson Correlation Between Features and Target
y = df_var['label']
lbl = y.copy()
df_var = df_var.drop(columns=['label'])
###try this###
pearson_thres = 0.01

high_corr_features = []
for feature in df_var.columns:
    # Drop NaN directly from feature and align target
    feature_clean = df_var[feature].dropna()
    target_clean = y[feature_clean.index]
    corr, _ = pearsonr(feature_clean, target_clean)
    if abs(corr) >= pearson_thres:  # Threshold for low correlation
        high_corr_features.append(feature)
# Keep only selected features
feature_pearson = df_var[high_corr_features]
df_pearson = pd.DataFrame(feature_pearson, columns=high_corr_features, index=feature_pearson.index)
print(f"Features with high correlation to target (|corr| >= {pearson_thres}): {len(high_corr_features)}, {len(df_pearson.columns)}")
df_pearson['label'] = lbl
df_pearson.to_csv('./dataset/MPData/tem/hf_manual_misrate_data_v6n.csv', index=False)

##Step 6.3: Chi-Square Test
y = df_pearson['label']
lbl = y.copy()
df_pearson = df_pearson.drop(columns=['label'])
###try this###
chi2_thres = 0.05
# Drop rows with NaN in feature_pearson
feature_clean = df_pearson.dropna()
target_clean = y[feature_clean.index]

chi2_selector = SelectKBest(score_func=chi2, k="all")
chi2_selector.fit(feature_clean, target_clean)
# Features with p-value < 0.05
chi2_features = [feature for feature, p_val in zip(df_pearson.columns, chi2_selector.pvalues_) if p_val < chi2_thres]
# Keep only selected features
feature_chi2 = df_pearson[chi2_features]
df_chi2 = pd.DataFrame(feature_chi2, columns=chi2_features, index=feature_chi2.index)
print(f"Features with p-value < {chi2_thres} by Chi2 test: {len(chi2_features)}, {len(df_chi2.columns)}")
df_chi2['label'] = lbl
df_chi2.to_csv('./dataset/MPData/tem/hf_manual_misrate_data_v7n.csv', index=False)

##Step 6.4: P-value >= 0.05 by ANOVA test
y = df_chi2['label']
lbl = y.copy()
df_chi2 = df_chi2.drop(columns=['label'])
###try this###
anova_thres = 0.05
# Drop rows with NaN in feature_pearson
feature_clean = df_chi2.dropna()
target_clean = y[feature_clean.index]

anova_selector = SelectKBest(score_func=f_classif, k="all")
anova_selector.fit(feature_clean, target_clean)
# Select features with p-value < 0.05
anova_features = [feature for feature, p_val in zip(df_chi2.columns, anova_selector.pvalues_) if p_val < anova_thres]
# Keep only selected features
feature_anova = df_chi2[anova_features]
df_anova = pd.DataFrame(feature_anova, columns=anova_features, index=feature_anova.index)
print(f"Selected features based on ANOVA (p < {anova_thres}): {len(anova_features)}, {len(df_anova.columns)}")
df_anova['label'] = lbl
df_anova.to_csv('./dataset/MPData/tem/hf_manual_misrate_data_v8n.csv', index=False)

# Step 7: Handle missing data
df_final = df_anova.copy()
# 7.1: Impute remaining missing values using KNN for numerical columns
numerical_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
print('numerical_cols', len(numerical_cols), len(df_final.columns))
if 'label' in numerical_cols:
    numerical_cols.remove('label')
imputer = KNNImputer(n_neighbors=3)
df_final[numerical_cols] = imputer.fit_transform(df_final[numerical_cols])

# 7.2: Apply mode imputation for categorical columns
categorical_cols = df_final.select_dtypes(include=["object"]).columns.tolist()
print('categorical_cols', categorical_cols)
for col in categorical_cols:
    df_final[col].fillna(df_final[col].mode()[0], inplace=True)
    
### Step 8: Save the final data ###
df_final['MRN_GNV'] = data_mis_df['MRN_GNV']
df_final.to_csv('./dataset/MPData/tem/hf_manual_misrate_data_strategy_strfinal.csv', index=False)

print(f'The final data has {len(df_final.columns)-1} features')