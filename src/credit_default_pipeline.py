"""
Credit default prediction pipeline.

Feature engineering (aggregation, rolling, recency, delinquency ratio, response
rates) -> XGBoost feature selection and grid search -> SHAP analysis ->
acceptance-threshold strategy -> neural network grid search.
"""

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# ============================================================================ #
# Configuration
# ============================================================================ #

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ML_PROJECT_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Source data
SAMPLED_TRAIN_DATA_PATH = DATA_DIR / "sampled_train_data.csv"

# Train / test splits
TRAIN_FEATURES_PATH = DATA_DIR / "train_df.csv"
TRAIN_TARGET_PATH = DATA_DIR / "train_y.csv"
TEST1_FEATURES_PATH = DATA_DIR / "test1_df.csv"
TEST1_TARGET_PATH = DATA_DIR / "test1_y.csv"
TEST2_FEATURES_PATH = DATA_DIR / "test2_df.csv"
TEST2_TARGET_PATH = DATA_DIR / "test2_y.csv"

# Model artifacts and results
FEATURE_IMPORTANCE_MODEL_1_PATH = DATA_DIR / "feature_importance_1.csv"
FEATURE_IMPORTANCE_MODEL_2_PATH = DATA_DIR / "feature_importance_2.csv"
XGB_GRID_SEARCH_RESULTS_PATH = DATA_DIR / "grid_search_results.csv"
XGB_MODEL_PATH = DATA_DIR / "xgb_model.pkl"
SHAP_SUMMARY_STATS_PATH = DATA_DIR / "Shap_Analysis.csv"
TRAIN_PD_SCORES_PATH = DATA_DIR / "train_df_PD.csv"
NN_GRID_SEARCH_RESULTS_PATH = DATA_DIR / "nn_grid_results.csv"

REFERENCE_DATE = pd.Timestamp("2018-03-31")
FILTER_END_DATE = "2018-04-30"

CATEGORICAL_COLS = [
    "B_30", "B_38", "D_114", "D_116", "D_117", "D_120",
    "D_126", "D_63", "D_64", "D_66", "D_68",
]

CUSTOMER_ID_COL = "customer_ID"
DATE_COL = "S_2"
SPEND_COL = "S_18"
BALANCE_COL = "B_15"

STRATEGY_START_DATE = "2017-10-01"
STRATEGY_END_DATE = "2018-03-31"


# ============================================================================ #
# Helper functions
# ============================================================================ #

def last_n_months_stats(df, months, num_columns):
    start_date = REFERENCE_DATE - pd.DateOffset(months=months)
    df_filtered = df[(df["S_2"] > start_date) & (df["S_2"] <= REFERENCE_DATE)]

    agg_df = df_filtered.groupby("customer_ID")[num_columns].agg(
        ["mean", "min", "max", "std"]
    )
    agg_df.columns = [f"{col}_{stat}_{months}m" for col, stat in agg_df.columns]

    return agg_df.reset_index()


def last_n_months_categorical_stats(df, months, cat_columns):
    start_date = REFERENCE_DATE - pd.DateOffset(months=months)
    df_filtered = df[(df["S_2"] > start_date) & (df["S_2"] <= REFERENCE_DATE)]

    response_rate = (
        df_filtered.groupby("customer_ID")[cat_columns].sum()
        / df_filtered.groupby("customer_ID")[cat_columns].count()
    )
    ever_response = df_filtered.groupby("customer_ID")[cat_columns].apply(
        lambda x: (x.sum() > 0).astype(int)
    )

    response_rate.columns = [
        f"{col}_Response_Rate_{months}m" for col in response_rate.columns
    ]
    ever_response.columns = [
        f"{col}_Ever_Response_{months}m" for col in ever_response.columns
    ]

    agg_df = response_rate.merge(ever_response, on="customer_ID")

    return agg_df.reset_index()


def calculate_default_and_revenue(
    df: pd.DataFrame,
    target_col: str,
    pd_col: str,
    balance_col: str,
    spend_col: str,
    threshold: float,
):
    accepted = df[df[pd_col] < threshold].copy()

    if accepted.empty:
        return 0.0, 0.0

    total_accepted = len(accepted)
    defaults = accepted[target_col].sum()
    default_rate = defaults / total_accepted

    accepted["B_Avg"] = accepted[
        [col for col in df.columns if balance_col in col]
    ].mean(axis=1)
    accepted["S_Avg"] = accepted[
        [col for col in df.columns if spend_col in col]
    ].mean(axis=1)

    accepted["MonthlyRevenue"] = accepted["B_Avg"] * 0.02 + accepted["S_Avg"] * 0.001

    accepted["ExpectedRevenue"] = accepted.apply(
        lambda row: 12 * row["MonthlyRevenue"] if row[target_col] == 0 else 0,
        axis=1,
    )

    total_expected_revenue = accepted["ExpectedRevenue"].sum()

    return default_rate, total_expected_revenue


def build_model_nn(n_hidden, n_nodes, activation, dropout, input_dim):
    model_nn = Sequential()
    model_nn.add(Dense(n_nodes, activation=activation, input_dim=input_dim))
    if dropout > 0:
        model_nn.add(Dropout(dropout))
    for _ in range(n_hidden - 1):
        model_nn.add(Dense(n_nodes, activation=activation))
        if dropout > 0:
            model_nn.add(Dropout(dropout))
    model_nn.add(Dense(1, activation="sigmoid"))
    model_nn.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[])
    return model_nn


# ============================================================================ #
# Step 1: Load and preprocess the data
# ============================================================================ #

sample_data = pd.read_csv(SAMPLED_TRAIN_DATA_PATH)
sample_data["S_2"] = pd.to_datetime(sample_data["S_2"])

print(
    "Columns with datetime dtype:",
    sample_data.select_dtypes(include=["datetime64"]).columns,
)


# ============================================================================ #
# Step 2: One-hot encoding
# ============================================================================ #

sample_data_encoded = pd.get_dummies(
    sample_data, columns=CATEGORICAL_COLS, drop_first=True, dtype=int
)

newly_created_columns = [
    col for col in sample_data_encoded.columns.tolist()
    if col not in sample_data.columns
]


# ============================================================================ #
# Step 3: Transactions filtering up to April 2018 (Cutoff date)
# ============================================================================ #

sample_data_filtered = sample_data[sample_data["S_2"] <= FILTER_END_DATE]
sample_data_filtered = sample_data_filtered.sort_values(by=["customer_ID", "S_2"])


# ============================================================================ #
# Step 4: Aggregation-based features
# ============================================================================ #

df = sample_data_filtered.copy()

num_features = [
    col for col in df.columns
    if col not in ["S_2", "customer_ID", "target"] + CATEGORICAL_COLS + newly_created_columns
]

aggregated_features = (
    df.set_index("customer_ID")
    .groupby("customer_ID")[num_features]
    .agg(["mean", "sum", "min", "max", "std"])
)
aggregated_features.columns = ["_".join(col) for col in aggregated_features.columns]

print(aggregated_features.head())
print(len(aggregated_features))


# ============================================================================ #
# Step 5: Rolling features
# ============================================================================ #

stats_3m = last_n_months_stats(df, 3, num_features)
stats_6m = last_n_months_stats(df, 6, num_features)
stats_9m = last_n_months_stats(df, 9, num_features)
stats_12m = last_n_months_stats(df, 12, num_features)

df_rolling = (
    stats_3m.merge(stats_6m, on="customer_ID", how="left")
    .merge(stats_9m, on="customer_ID", how="left")
    .merge(stats_12m, on="customer_ID", how="left")
)
print(df_rolling.head())

final_file = pd.merge(aggregated_features, df_rolling, on="customer_ID", how="inner")
print(len(final_file))
print(final_file.head())


# ============================================================================ #
# Step 6: Recency-based features
# ============================================================================ #

last_statement_date = df.groupby("customer_ID")["S_2"].max()

days_since_last_statement = (
    (REFERENCE_DATE - last_statement_date)
    .dt.days
    .reset_index()
    .add_suffix("_days_since_last_statement")
    .rename(columns={"customer_ID_days_since_last_statement": "customer_ID"})
)

print(days_since_last_statement.head())
print(len(days_since_last_statement))

final_file = final_file.merge(days_since_last_statement, on="customer_ID", how="inner")
print(final_file.head())


# ============================================================================ #
# Step 7: Delinquency-risk ratio
# ============================================================================ #

d_col = df.filter(regex="^D_").select_dtypes(include=[np.number]).columns
r_col = df.filter(regex="^R_").select_dtypes(include=[np.number]).columns

df_grouped = df.groupby("customer_ID")[list(d_col)].mean().reset_index()
df_grouped["D_avg"] = df_grouped[d_col].mean(axis=1)
df_grouped["R_avg"] = (
    df.groupby("customer_ID")[list(r_col)].mean().reset_index()[r_col].mean(axis=1)
)

print(df_grouped["R_avg"].head())
print(df_grouped["D_avg"].head())
print(df_grouped.head())

final_file = final_file.merge(
    df_grouped[["customer_ID", "D_avg", "R_avg"]], on="customer_ID", how="left"
)
print(final_file.head())

final_file["D_To_R_Ratio"] = final_file["D_avg"] / final_file["R_avg"]
print(final_file.head())

final_file_copy = final_file.copy()


# ============================================================================ #
# Step 8: Response rate features
# ============================================================================ #

stats_3m = last_n_months_categorical_stats(sample_data_encoded, 3, newly_created_columns)
stats_6m = last_n_months_categorical_stats(sample_data_encoded, 6, newly_created_columns)
stats_9m = last_n_months_categorical_stats(sample_data_encoded, 9, newly_created_columns)
stats_12m = last_n_months_categorical_stats(sample_data_encoded, 12, newly_created_columns)

df_response = (
    stats_3m.merge(stats_6m, on="customer_ID", how="left")
    .merge(stats_9m, on="customer_ID", how="left")
    .merge(stats_12m, on="customer_ID", how="left")
)

print(df_response["D_68_6.0_Response_Rate_12m"].describe())
print(len(df_response))

final_file1 = pd.merge(final_file, df_response, on="customer_ID", how="inner")
final_file1 = final_file1.replace({None: np.nan})

df1 = df.drop_duplicates(subset="customer_ID")
final_file1 = final_file1.merge(df1[["customer_ID", "target"]], on="customer_ID", how="left")
final_file1 = final_file1.drop(columns="customer_ID")


# ============================================================================ #
# Step 9: Train / test split
# ============================================================================ #

train_df, temp_df = train_test_split(final_file1, test_size=0.3, random_state=42)
test1_df, test2_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(len(train_df))
print(len(test1_df))
print(len(test2_df))

train_y = train_df["target"]
train_df = train_df.drop(columns="target")

test1_y = test1_df["target"]
test1_df = test1_df.drop(columns="target")

test2_y = test2_df["target"]
test2_df = test2_df.drop(columns="target")


# ============================================================================ #
# Step 10: Feature selection via two XGBoost models
# ============================================================================ #

model_1 = xgb.XGBClassifier()
model_1.fit(train_df, train_y)

importances_1 = model_1.feature_importances_
feature_importance_df_1 = pd.DataFrame({
    "feature": train_df.columns,
    "importance": importances_1,
})
feature_importance_df_1 = feature_importance_df_1.sort_values(
    by="importance", ascending=False
)
feature_importance_df_1.to_csv(FEATURE_IMPORTANCE_MODEL_1_PATH, index=False)

model_2 = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.5,
    max_depth=4,
    subsample=0.5,
    colsample_bytree=0.5,
    scale_pos_weight=5,
    missing=np.nan,
    use_label_encoder=False,
)
model_2.fit(train_df, train_y)

importances_2 = model_2.feature_importances_
feature_importance_df_2 = pd.DataFrame({
    "feature": train_df.columns,
    "importance": importances_2,
})
feature_importance_df_2 = feature_importance_df_2.sort_values(
    by="importance", ascending=False
)
feature_importance_df_2.to_csv(FEATURE_IMPORTANCE_MODEL_2_PATH, index=False)

combined_importance = pd.concat(
    [
        feature_importance_df_1.set_index("feature"),
        feature_importance_df_2.set_index("feature"),
    ],
    axis=1,
    keys=["model_1", "model_2"],
)
combined_importance["max_importance"] = combined_importance.max(axis=1)

important_features = combined_importance[
    combined_importance["max_importance"] > 0.005
].index.tolist()

print(len(important_features))
print(important_features)

train_df = train_df[important_features]
test1_df = test1_df[important_features]
test2_df = test2_df[important_features]

train_df.to_csv(TRAIN_FEATURES_PATH, index=False)
test1_df.to_csv(TEST1_FEATURES_PATH, index=False)
train_y.to_csv(TRAIN_TARGET_PATH, index=False)
test1_y.to_csv(TEST1_TARGET_PATH, index=False)
test2_y.to_csv(TEST2_TARGET_PATH, index=False)


# ============================================================================ #
# Step 11: XGBoost hyperparameter grid search
# ============================================================================ #

results = []

for n_estimators in [50, 100, 300]:
    for learning_rate in [0.01, 0.1]:
        for subsample in [0.5, 0.8]:
            for colsample_bytree in [0.5, 1.0]:
                for scale_pos_weight in [1, 5, 10]:

                    model = xgb.XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        scale_pos_weight=scale_pos_weight,
                        missing=np.nan,
                        use_label_encoder=False,
                    )
                    model.fit(train_df, train_y)

                    y_train_pred = model.predict_proba(train_df)[:, 1]
                    y_test1_pred = model.predict_proba(test1_df)[:, 1]
                    y_test2_pred = model.predict_proba(test2_df)[:, 1]

                    auc_train = roc_auc_score(train_y, y_train_pred)
                    auc_test1 = roc_auc_score(test1_y, y_test1_pred)
                    auc_test2 = roc_auc_score(test2_y, y_test2_pred)

                    results.append({
                        "Trees": n_estimators,
                        "LR": learning_rate,
                        "Subsample": f"{int(subsample * 100)}%",
                        "% Features": f"{int(colsample_bytree * 100)}%",
                        "Weight of Default": scale_pos_weight,
                        "AUC Train": auc_train,
                        "AUC Test 1": auc_test1,
                        "AUC Test 2": auc_test2,
                    })

                    # Written every iteration to avoid losing progress on a crash
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(XGB_GRID_SEARCH_RESULTS_PATH, index=False)

grid_search_results = pd.read_csv(XGB_GRID_SEARCH_RESULTS_PATH)


# ============================================================================ #
# Step 12: Bias / stability check
# ============================================================================ #

grid_search_results["Avg AUC Test"] = grid_search_results[
    ["AUC Test 1", "AUC Test 2"]
].mean(axis=1)
grid_search_results["AUC Test Variance"] = grid_search_results[
    ["AUC Train", "AUC Test 1", "AUC Test 2"]
].var(axis=1)

best_model_variance = grid_search_results.loc[
    grid_search_results["Avg AUC Test"].idxmax(), "AUC Test Variance"
]

top_models = grid_search_results.nlargest(5, "Avg AUC Test")
most_stable_model = top_models.loc[top_models["AUC Test Variance"].idxmin()]

print(most_stable_model)


# ============================================================================ #
# Step 13: Final XGBoost model
# ============================================================================ #

model_3 = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.01,
    subsample=0.5,
    colsample_bytree=1,
    scale_pos_weight=1,
    missing=np.nan,
    use_label_encoder=False,
)
model_3.fit(train_df, train_y)

joblib.dump(model, XGB_MODEL_PATH)


# ============================================================================ #
# Step 14: SHAP analysis
# ============================================================================ #

explainer = shap.Explainer(model_3)
shap_values = explainer(train_df)

shap_importance = pd.DataFrame({
    "Feature": train_df.columns,
    "Mean SHAP": abs(shap_values.values).mean(axis=0),
})
shap_importance = shap_importance.sort_values(by="Mean SHAP", ascending=False)

top_5_features = shap_importance.head(5)
print(top_5_features)

top_5_feature_names = top_5_features["Feature"].tolist()
top_5_data = train_df[top_5_feature_names]
print(top_5_data.head())

summary_statistics = top_5_data.describe(percentiles=[0.01, 0.05, 0.95, 0.99])
summary_statistics.loc["% Missing"] = top_5_data.isnull().mean() * 100

print("Summary Statistics for Top 5 Features with Highest SHAP Values (including % missing):")
print(summary_statistics)

summary_statistics.to_csv(SHAP_SUMMARY_STATS_PATH)


# ============================================================================ #
# Step 15: SHAP plots
# ============================================================================ #

model = joblib.load(XGB_MODEL_PATH)
test2_df = pd.read_csv(TEST2_FEATURES_PATH)

explainer = shap.Explainer(model)
shap_values_test2 = explainer(test2_df)

shap.plots.beeswarm(shap_values_test2, max_display=20)
shap.plots.waterfall(shap_values_test2[8])


# ============================================================================ #
# Step 16: Strategy - acceptance threshold
# ============================================================================ #

pd_values = model.predict_proba(train_df)[:, 1]
train_df["Y_Hat"] = pd_values
train_df.to_csv(TRAIN_PD_SCORES_PATH, index=False)

df[DATE_COL] = pd.to_datetime(df[DATE_COL])

df_6m = df[
    (df[DATE_COL] >= STRATEGY_START_DATE) & (df[DATE_COL] <= STRATEGY_END_DATE)
].copy()

avg_df = df_6m.groupby(CUSTOMER_ID_COL)[[SPEND_COL, BALANCE_COL]].mean().reset_index()
avg_df = avg_df.rename(columns={SPEND_COL: "S_Avg", BALANCE_COL: "B_Avg"})

new_cols = pd.concat([train_y, train_df["Y_Hat"]], axis=1)
print(new_cols.head())

df_merged = pd.concat([avg_df, new_cols], axis=1)
df_merged.drop(columns="customer_ID")
print(df_merged.head())

print(calculate_default_and_revenue(df_merged, "target", "Y_Hat", "B_Avg", "S_Avg", 0.65))

# Aggressive threshold = 0.93 | Conservative threshold = 0.6


# ============================================================================ #
# Step 17: Neural network grid search
# ============================================================================ #

imputer = SimpleImputer(strategy="constant", fill_value=0)
X_train_imputed = imputer.fit_transform(train_df)
X_test1_imputed = imputer.transform(test1_df)
X_test2_imputed = imputer.transform(test2_df)

nn_hidden_layers_list = [2, 4]
nn_nodes_list = [4, 6]
nn_activation_list = ["relu", "tanh"]
nn_dropout_list = [0.5, 0.0]
nn_batch_size_list = [100, 10000]
nn_epochs = 20

if os.path.exists(NN_GRID_SEARCH_RESULTS_PATH):
    result_df_nn = pd.read_csv(NN_GRID_SEARCH_RESULTS_PATH)
else:
    result_df_nn = pd.DataFrame(columns=[
        "HL", "# Node", "Activation Function", "Dropout", "Batch Size",
        "AUC Train", "AUC Test1", "AUC Test2",
    ])

for nn_hl in nn_hidden_layers_list:
    for nn_node in nn_nodes_list:
        for nn_act in nn_activation_list:
            for nn_drop in nn_dropout_list:
                for nn_bs in nn_batch_size_list:
                    # ... training & evaluation code ...

                    try:
                        auc_train_nn = roc_auc_score(
                            train_y, model_nn.predict(X_train_imputed).ravel()
                        )
                        auc_test1_nn = roc_auc_score(
                            test1_y, model_nn.predict(X_test1_imputed).ravel()
                        )
                        auc_test2_nn = roc_auc_score(
                            test2_y, model_nn.predict(X_test2_imputed).ravel()
                        )
                    except Exception as e:
                        print(f"Error in AUC calculation: {e}")
                        auc_train_nn, auc_test1_nn, auc_test2_nn = np.nan, np.nan, np.nan

                    new_row_nn = pd.DataFrame([{
                        "HL": nn_hl,
                        "# Node": nn_node,
                        "Activation Function": nn_act,
                        "Dropout": f"{int(nn_drop * 100)}%",
                        "Batch Size": nn_bs,
                        "AUC Train": auc_train_nn,
                        "AUC Test1": auc_test1_nn,
                        "AUC Test2": auc_test2_nn,
                    }])

                    result_df_nn = pd.concat([result_df_nn, new_row_nn], ignore_index=True)

result_df_nn.to_csv(NN_GRID_SEARCH_RESULTS_PATH, index=False)
print("NN Grid Search Complete. Results saved to:", NN_GRID_SEARCH_RESULTS_PATH)

nn_grid_results = pd.read_csv(NN_GRID_SEARCH_RESULTS_PATH)


# ============================================================================ #
# Step 18: Neural network stability check
# ============================================================================ #

nn_grid_results["Avg AUC Test"] = nn_grid_results[["AUC Test1", "AUC Test2"]].mean(axis=1)
nn_grid_results["AUC Test Variance"] = nn_grid_results[
    ["AUC Train", "AUC Test1", "AUC Test2"]
].var(axis=1)

nn_best_model_variance = nn_grid_results.loc[
    nn_grid_results["Avg AUC Test"].idxmax(), "AUC Test Variance"
]

nn_top_models = nn_grid_results.nlargest(5, "Avg AUC Test")
nn_most_stable_model = nn_top_models.loc[nn_top_models["AUC Test Variance"].idxmin()]

print(nn_most_stable_model)
