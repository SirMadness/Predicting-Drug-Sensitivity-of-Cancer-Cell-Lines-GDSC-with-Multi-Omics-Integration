#!/usr/bin/python

# Copyright Brian Rabkin All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Imports
import numpy as np
import pandas as pd
import polars as pl, pyarrow # required for polars file reading

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
from   plotly.subplots import make_subplots

import tabulate

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from scipy.stats import pearsonr, spearmanr

# globals
ROUND_TO = 3

# read parque to pd.DataFrame
def read_parque_to_pd_df(file_path):
    "read parque with polars and then convert to pandas for speed"
    if file_path.exists():
        dl_out = pl.read_parquet(file_path)
        #print("Loaded dataframe shape:",dl_out.shape)
        df_out = dl_out.to_pandas()
        return df_out
    else:
        print(f"File not found: {file_path}\nPlease run the methylation ingestion and transpose step first.")
        return None

def print_dropped_row_info(dict_dropped_rows):
    """
    Print information captured throught the EDA process for why rows were dropped.
    Args: NONE  uses global dict_dropped_rows
    Ouput: Prints information

    """
    
    print("List of reasons rows removed:")
    remaining_rows = 0
    for key, value in dict_dropped_rows.items():
        if key == 'Start':
            print(f"\tStarted with \t\t{value:,.0f} rows\n\t---------------")
            start_total = value
            remaining_rows = start_total
            continue
        print(f"\t{key}: \t{value:,.0f} rows\t {(value/start_total)*100:.2f}% removed")
        remaining_rows = remaining_rows - value
    print(f"\t---------------\n\tRemaining Rows: \t{remaining_rows:,.0f}\t {(remaining_rows/start_total)*100:,.2f}% of original")

def print_dropped_dic_features(dict_features):
    """
    Print information cpatured throught the EDA process for why features were droppped
    Args: None, uses globarl dict_features
    
    """
    print("List of dropped features:")
    print("Feature:\tReason:")
    for key, value in dict_features.items():
        print(f"{key}:\t\t{value}")

        # capture upper, lower bounds and iqr from features

def count_decimal_places(f):
    s = format(f, 'f').rstrip('0')
    if '.' in s:
        return len(s.split('.')[-1])
    return 0

def calculate_bounds(df,feature, df_bounds):
    '''
    calculate_bounds calculates upper, lower bounds and iqr
    Updates df_bounds with a new row (feature, lower, upper, q1, q2, iqr)
    
    Args:
        df (dataframe):           dataframe
        feature (df column name): column name to claculate values from
        df_bounds (dataframe): dataframe with the output of this definition
    returns:
        df_bounds 
        '''
    
    row = []
    columns = ['feature', 'q1', 'q3', 'iqr']
    print(f"{feature} bounds:")
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    if df[feature].min() >= 0: # avoid negative bound for data that is not negative (mpg for example)
        lower_bound = 0
    else:
        lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    pprecision = max(count_decimal_places(lower_bound), count_decimal_places(upper_bound))
   
    print(f"\tLower_bound: {lower_bound:,.{pprecision}f}\n\tUpper_bound: {upper_bound:,.{pprecision}f}")

    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    print(f"\tOutliers detected: {len(outliers):,.0f} of {len(df):,.0f}: {len(outliers)/len(df)*100:.2f}%")

    # check if the outliers are brand specific
    #df[df[feature] > upper_bound].groupby(['manufacturer','model', 'year'])[feature].describe()

    # For each model, what is the price range and age
    #print(f"Note that the upper bound is very low for expensive cars at ${upper_bound:,.0f}")

    row = {
        'feature': feature,
        'lower': lower_bound,
        'upper': upper_bound,
        'q1': q1,
        'q3': q3,
        'iqr': iqr
    }
    if feature in df_bounds['feature'].values:
        idx = df_bounds[df_bounds['feature'] == feature].index[0]
        for k, v in row.items():
            df_bounds.at[idx, k] = v
    else:
        df_bounds.loc[len(df_bounds)] = row # should fix this so that it does not add another row per feature

    return(df_bounds)

# Are two columns effectively the same? 
def are_columns_effectively_the_same(df_col0, df_col1, verbose=True):
    """ updated 20250628
    Function to check if two columns are effectively the same
    Are all vallues in col1 mapped 1:1 to col2? 
    Skipping 'nan' values
    Args:
        df[[col1]] (pd.DataFrame): DataFrame with First column to check
        df[[col2]] (pd.DataFrame): DataFrame with Second column to check
        verbose (bool):  Prints out which values in col0 map to multiple values in col1
    Returns:
        bool: True if columns are the same, False otherwise
    """
    col0 = df_col0.columns[0]
    col1 = df_col1.columns[0]
    nunique_0 = df_col0[col0].nunique()
    nunique_1 = df_col1[col1].nunique()
    print(f"Num Unique {col0}:\t{nunique_0}: Num null {df_col0[col0].isnull().sum()}")
    print(f"Num Unique {col1}:\t{nunique_1}: Num null {df_col1[col1].isnull().sum()}")
    matching = True # track if they match
    n_missmatches = 0
    if nunique_0 != nunique_1:
        print("NOT THE SAME: There is a different # of nunique between columns\n")
        return False
    
    for unique_val in df_col0[col0].unique():
        if unique_val == 'nan':
            continue
        
        if df_col1[col1][df_col0[col0] == unique_val].nunique() > 1:
            matching = False
            print('here')
            n_missmatches += 1
            if verbose:
                print(f"Column {col0} has value {unique_val} that maps to multiple values in column {col1}\n")

    if matching:
        print("1:1 match for all unique values in column 1 with column 2\n")   
        return True
    else:
        print(f"{n_missmatches} features in Col0 do not match in Col1")
        return False

# Determine the precent of data missing and drop the columns with more than 50% missing data
def remove_columns_with_missing_data(df, perc_cuttoff=50):
    """ 
    Function to remove columns with more than perc_cuttoff% missing data
    Args:
        df (pd.DataFrame): DataFrame to check for missing data
        perc_cuttoff (int): Percentage cutoff for dropping columns
    Returns:
        pd.DataFrame: DataFrame with columns dropped
    Hint:
        to suppress print of df, use "_ = remove_columns_with_missing_data()"
    """
    #qc_perc_null = df.isnull().sum()/len(df)*100
    #print(f"\nPercentage of missing data:\n{qc_perc_null.sort_values(ascending=False):'%'f}\n")

    qc_perc_null = df.isnull().sum().div(len(df)).mul(100)

    out = (
        qc_perc_null
        .sort_values(ascending=False)
        .map(lambda x: f"{x:.2f}%")      # 2-decimal percentage with % sign
    )

   
    print(f"There are a total of {len(df)} rows with the following percentage of missing data:")
    print(out)
    #print(f"Drop columns:\n{qc_perc_null[qc_perc_null > 50]}")
    if len(qc_perc_null[qc_perc_null > perc_cuttoff]) == 0:
        print(f"\nNo columns with >{perc_cuttoff}% missing data to drop")
    else:
        print(f"\nDropped Columns with >{perc_cuttoff}% missing data:")
        for col, val in qc_perc_null[qc_perc_null > perc_cuttoff].items():
            print(f"\t{col}: {val:.2f}%\n")
            dict_features[col] = f"had > 50% missing rows"

    df = df.drop(columns = qc_perc_null[qc_perc_null >= perc_cuttoff].index)
    return df

def get_rows_with_nulls_in_common(df, columns):
    '''
    Get the isnull row indices in common between columns
    Args:
        df (dataframe):           dataframe
        columns (df column name): list of column names to compare
    returns:
        idx: List of rows indices from the input columns in which the are all null
    '''
    
    idx = []
    for col in columns:
        if len(idx) > 0:
            idx = list(set(idx) & set(df[df[col].isnull()].index.tolist()))
        else:
            idx = df[df[col].isnull()].index.tolist()
    return idx


# Explore missing data and strings with 'nan', 'na', 'unknown'
def print_unique_missing_values(df, print_stats=True, print_unique=False, sort_vals=False):
    """
    Function to review unique values in columns.
    Prints the unique values, number of unique values, number of missing values, percentage of missing values, and number of 'nan' values in each column.
    Args:
        df (pd.DataFrame): DataFrame to check for unique values
        printtxt (boolt): Prints the text output of missing values. (defualt=True)
        print_unique (bool): Print unique values (default=False)
        sort_vals (bool): Sort the unique values to be printed (default=False)
    Returns:
        pd.DataFrame: DataFrame with summary statistics for object columns
    """
    num_rows = len(df) # rows in df
    data = [] # for building df_unknown
    data.append({
        'feature': 'Total',
        'notnull_num': num_rows,
        'notnull_percent': 100,
        'unique_num': 0,
        'unique__avg_rows_per_unique': 0.00,
        'null_num': 0,
        'null_percent': 0.00,
        'nan_str_num': 0,
        'nan_str_percent': 0.00,
        'na_str_num': 0,
        'na_str_percent': 0.00,
        'missing_str_num': 0,
        'missing_str_percent': 0.00,
        'unknown_str_num': 0,
        'unknown_str_percent': 0.00
    })
    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        col_series = df[col].astype(str)
        notnull_num = col_series.notnull().sum()
        unique_num = col_series.nunique(dropna=True)
        unique__avg_rows_per_unique = round(notnull_num / unique_num if unique_num else 0.00, 2)
        null_num = col_series.isnull().sum()
        nan_str_num = (col_series.str.lower() == 'nan').sum()
        na_str_num = (col_series.str.lower() == 'na').sum()
        missing_str_num = (col_series.str.lower() == 'missing').sum()
        unknown_str_num = (col_series.str.lower() == 'unknown').sum()
        data.append({
            'feature': col,
            'notnull_num': notnull_num,
            'notnull_percent': round((notnull_num / num_rows) * 100, 2),
            'unique_num': unique_num,
            'unique__avg_rows_per_unique': unique__avg_rows_per_unique,
            'null_num': null_num,
            'null_percent': round((null_num / num_rows) * 100, 2),
            'nan_str_num': nan_str_num,
            'nan_str_percent': round((null_num / num_rows) * 100, 2),
            'na_str_num': na_str_num,
            'na_str_percent': round((na_str_num / num_rows) * 100,2),
            'missing_str_num': missing_str_num,
            'missing_str_percent': round((missing_str_num / num_rows) * 100,2),
            'unknown_str_num': unknown_str_num,
            'unknown_str_percent': round((unknown_str_num / num_rows) * 100,2)
        })
        if print_stats:
            print(f"{col}")
            print(f"Number of rows in df          : {num_rows:,.0f} rows")
            print(f"Number of unique values       : {unique_num:,.0f}\t(Average of {unique__avg_rows_per_unique:,.0f} non-nan entries/unique value)")
            print(f"Number of null    values      : {null_num:,.0f}\t({(null_num/num_rows)*100:.2f}%)")
            print(f"Number of 'nan' str value     : {nan_str_num}\t({(nan_str_num/num_rows)*100:.2f}%)")
            print(f"Number of 'na' str value      : {na_str_num}\t({(na_str_num/num_rows)*100:.2f}%)")
            print(f"Number of 'missing' str value : {missing_str_num}\t({(missing_str_num/num_rows)*100:.2f}%)")
            print(f"Number of 'unknown' str value : {unknown_str_num}\t({(unknown_str_num/num_rows)*100:.2f}%)\n")
        
        if print_unique:
            if sort_vals:
                list_out = sorted(col_series.dropna().unique())
            else:
                list_out = col_series.dropna().unique()
            vals_per_line = 5
            print(
                f"Unique values in '{col}':\n\t" +
                '\n\t'.join([', '.join(map(str, list_out[i:i + vals_per_line]))
                for i in range(0, len(list_out), vals_per_line)])
            )
            print("\n")
    df_unknown = pd.DataFrame(data)
    return df_unknown

# 
def make_lookup_df_for_matched_columns(df_col0, df_col1):
    """ updated 20250628
    Function to make a lookup dataframe for effectively identical columns so one can be dropped.
    
    Args:
        df_col0 (pd.DataFrame): DataFrame with single column
        df_col1 (pd.DataFrame): DataFrame with single column
    Returns:
        bool: True if columns are the same, False otherwise
        dataframe: Dataframe with two columns made of matching unique values for each original column
    """
    col0 = df_col0.columns[0]
    col1 = df_col1.columns[0]
    nunique_0 = df_col0[col0].nunique()
    nunique_1 = df_col1[col1].nunique()
    print(f"Num Unique {col0}: {nunique_0}: Num nul {df_col0[col0].isnull().sum()}")
    print(f"Num Unique {col1}: {nunique_1}: Num nul {df_col1[col1].isnull().sum()}")

    # Should a check be implemented? or make the lookup eather way?
    # Check to see if there is a 1:1 relationship (while mutting print out)
    f = StringIO()
    with redirect_stdout(f):
        same_q = are_columns_effectively_the_same(df_col0, df_col1)

    if not same_q:
        print("NOT THE SAME: There is a different # of nunique between columns\n")
        return None
    
    pairs = []
    max_len = 0
    for unique_val in df_col0[col0].unique():

        matched_names = df_col1.loc[df_col0[col0] == unique_val, col1].unique()
        if max_len < len(matched_names): # check it is a 1:1 relationship
            max_len = len(matched_names)

        if max_len > 1: # confirm there is not a one to many relationship
            print("NOT THE SAME: There is a different # of nunique between columns\n")
            return None
        
        for name in matched_names:
            pairs.append((unique_val, name))


    # Convert to DataFrame for easy viewing
    matched_pairs_df = pd.DataFrame(pairs, columns=[col0, col1])
    return matched_pairs_df



## -- Printing Functions --
# make printing markdown of dataframe easier
import tabulate
def print_df_markdown(df):
    print(df.to_markdown(index=False))


## -- Plotting Functions --
def boxplot_columns(df, columns):
    ''' 
    Generate box plots for the columns provided 

    args:
        df: dataframe
        columns: list of columns to plot

    return: (nothing returned)
    '''
    for col in columns:
        plt.figure(figsize=(8,4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()


def corr_matrix_plot(df, title_str='Correlation Heatmap'):
    """ Standardized correlation matrix plot with title
        Args:
            df (dataframe): dataframe with only the columns to correlate
            title_str (str): Opti
    
    """
    corr_m = (
        df            # keep only the three numeric columns
            .astype("float64") # cast in case any are stored as object
            .corr()            # default = Pearson correlation
    )

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_m, annot=True, cmap='coolwarm')
    plt.title(title_str)
    plt.show()
    return corr_m

def plot_treemap(tmp_df, feature_list, title=None):
    # make treemap of specific drugs with duplicate ids

    group_df = tmp_df.groupby(feature_list).size().reset_index(name='count')

    fig = px.treemap(group_df,
                    path=feature_list,
                    values='count',
                    color='count',
                    color_continuous_scale='Blues',
                    title=title
                    )
    #fig.update_layout(width=1000, height=800)
    #fig.show()
    return fig

def plot_sunburst(df, feature_list, title=None, color_continuous_scale='Viridis'):
    # Prepare data for sunburst chart
    df_grouped = df.groupby(feature_list).size().reset_index(name='count')

    #first_feature = feature_list[0]
    #df_grouped = df_grouped.sort_values(first_feature)
    
    df_grouped = (
        df
        .groupby(feature_list)
        .size()
        .reset_index(name='count')
        .sort_values(feature_list[0])
    )
    

    # Create sunburst chart using plotly express

    fig = px.sunburst(
        df_grouped, 
        path=feature_list,
        values='count',
        color='count',
        color_continuous_scale=color_continuous_scale,
        title= title
    )
    fig.update_traces(sort=False, selector=dict(type='sunburst'))
    fig.update_layout(width=1000, height=1000)
    #fig.show()
    return fig


def get_number_events(df, feature1, feature2):
    """ 
    How many unique values of feature 2 are in each of feature 1?

    args:
        df (dataframe): dataframe to be analyzed
        feature1 (str): primary feature with feature2 as subsets of feature1
        feature2 (str): feature to be counted for each unique value in feature1

    return:
        counts_df (dataframe): dataframe with counts of feature2 for each unique object of feature1
    """
    item_counts = df.groupby(feature1)[feature2].nunique()
    counts_df = (
        item_counts
        .reset_index(name='num_items')
    )
    return(counts_df)

## COPY FROM PROMPT III
## BAR TODO move to ml_utils.py and fix in PromptIII

# model metrics for evaluation
def plot_pr_auc(fig_title, precision, recall, pr_auc):
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def inference_data_type(target_series, frac_thresh=0.5):
    """
    args: 
      target_series (pd.series): the target data

    returns:
      (str): 'regression' if continuous, 'classification' otherwise
    """
    import pandas as pd
    n, u = len(target_series), target_series.nunique()

    if pd.api.types.is_numeric_dtype(target_series) and (u/n) > frac_thresh:
        return 'regression'
    else:
        return 'classification'

from sklearn.metrics import classification_report
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, fbeta_score,
    precision_recall_curve, average_precision_score, ConfusionMatrixDisplay
)

# Define Evaluation Metrics Function
def get_model_evaluation_metrics(y_true, y_pred, sample_weights=None, print_results=False, model_name="Model", ):
    """
    Calculates and prints key regression metrics RMSE, MAE, R-squared, Pearson Corr, Spearman Corr
    
    Args:
        y_true (series): true values for y
        y_pred (series): predicted values for y
        sample_weights (series): default=none, sample weights for y if the model used weights during trianing
        print_results (bool): True = print, False=default = no printing

    output:
        dictionary with results
    """
    # Accept both None and np.ndarray for sample_weights
    if type(sample_weights) is None:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
    else:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weights))
        mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weights)
        r2 = r2_score(y_true, y_pred, sample_weight=sample_weights)

    pearson, _ = pearsonr(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)

    if print_results:
        print(f"\n--- {model_name} Performance ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        print(f"Pearson Correlation: {pearson:.4f}")
        print(f"Spearman Correlation: {spearman:.4f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Pearson": pearson, "Spearman": spearman}


# function for turning regression predictions into binary predictions
def convert_reg_pred_to_binary(df, y_pred, threshholds):
#   • regression ⇒ class  (hard)
    df["binary_reg_pred"] = (
        y_pred < df["drug_id"].map(threshholds)
    ).astype(int)

    return df["binary_reg_pred"]

# Add classification metrics to regression model
def add_binary_metrics(
    df_models_results: pd.DataFrame, # dataframe with model results
    model_name: str,  # model name to match in df_models_results
    split_name: str, # split name to match in df_models_results
    df: pd.DataFrame, # dataframe with predictions,
    y_true: pd.Series,
    y_pred: pd.Series,
    #y_score: pd.Series = None,
    threshold_per_drug=None,  # dict with drug_id as key and threshold as value
    pos_label=None,
    round_to: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) Creates a binary prediction column in `df` using
       convert_reg_pred_to_binary(...) and `threshold_per_drug`.
    2) Computes classification metrics (accuracy, PR-AUC, precision, recall, fbeta)
       and writes them into the single row of `df_models` matching
       (Model == model_name) & (Dataset == split_name).
    3) Returns (df_models, df) with updates applied.
    """

    # — 1) Make the binary labels in df —
    bin_col = f"{model_name}_y_pred_bin"
    df[bin_col] = convert_reg_pred_to_binary(
        df,
        y_pred,
        threshold_per_drug
    )
    y_pred_bin = df[bin_col]

    # — 2) Locate the row in df_models to update —
    mask = (
        (df_models_results['Model'] == model_name) &
        (df_models_results['Split'] == split_name)
    )
    if not mask.any():
        raise ValueError(f"No row for model={model_name}, dataset={split_name}")

    # — 3) Compute metrics —
    acc = accuracy_score(y_true, y_pred_bin)
    # pr_auc = (
    #     average_precision_score(y_true, y_score)
    #     if y_score is not None else np.nan
    # )
    prec = precision_score(
        y_true, y_pred_bin,
        pos_label=pos_label,
        zero_division=0
    )
    rec = recall_score(
        y_true, y_pred_bin,
        pos_label=pos_label,
        zero_division=0
    )
    f1  = fbeta_score(
        y_true, y_pred_bin,
        pos_label=pos_label,
        beta=1,
        zero_division=0
    )

    # — 4) Round & assign back via .loc (no chained indexing) —
    metrics = {
        'Accuracy':  round(acc, round_to),
        # 'pr_auc':    round(pr_auc, round_to) if not np.isnan(pr_auc) else np.nan,
        'Precision': round(prec, round_to),
        'Recall':    round(rec, round_to),
        'FBeta':     round(f1, round_to),
    }

    # first, pull out the single integer index of the row to update
    idx = df_models_results.index[mask][0]

    # now assign only into columns that already exist
    for col, val in metrics.items():
        if col not in df_models_results.columns:
            raise KeyError(f"Column `{col}` not found in df_models_results; did you mean a different name?")
        df_models_results.at[idx, col] = val

    return df_models_results, df


## BAR valuate_model was updated after 16.1.valuate_model_new
def evaluate_model(model=None, model_name=None, data_split=None, X=None, y=None, sample_weights=None, df_models_eval=None,
                    train_time=None, pos_label=None, round_to=ROUND_TO, threshhold_per_drug=None, plot_pr_auc=False):
    """
    Evaluates a classification model and returns a dictionary of metrics.

    Args:
        model: Trained model or pipeline with predict/predict_proba.
        model_name (str): Name of the model.
        X (pd.DataFrame): Features.
        y (pd.Series or np.array): True labels.
        data_split (str): 'Train' or 'Test'.
        train_time (float, optional): Training time in seconds.
        pos_label (str or int): Positive class label.
        round_to (int): rounding number
        plot_pr_auc (bool): plot the Precision-Recall AUC

    Returns:
        dict: Row of metrics for appending to results DataFrame.
    """
    import time
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr, spearmanr

    cols_df_model = [
        'Model','Split','Model_Type','Time','Accuracy','Precision','Recall',
        'FBeta','PR_AUC','RMSE','MAE','R2','Pearson','Spearman'
    ]
    # initialise df_models_eval if first call
    if df_models_eval is None:
        # Initialize results DataFrame
        df_models_eval = pd.DataFrame(columns=cols_df_model)

    if model is None:
        return df_models_eval

    if data_split not in ['Train', 'Test']:
        print("data_split should be 'Train' or 'Test'")
        return _
    
    # Predict labels and probabilities
    start_pred = time.time()
    y_pred = model.predict(X)
    pred_time = time.time() - start_pred

    if data_split == 'Train':
        delta_time = train_time
    else: 
        delta_time =  pred_time

    # get metrics for the  model
    reggres_metrics = get_model_evaluation_metrics(y, y_pred, sample_weights)
    
    # check if regression (continuous data)
    model_type = inference_data_type(y)

    if model_type  == 'regression':
        ## PROCESS REGRESSION MODEL TYPE

        row = {
            'Model': model_name,
            'Model_Type': model_type,
            'Split': data_split,
            'Time':       round(delta_time, round_to),
            'Accuracy':   np.nan, #round(accuracy, ROUND_TO),
            'Precision':  np.nan, #round(precision, ROUND_TO),
            'Recall':     np.nan, #round(recall, ROUND_TO),
            'FBeta':      np.nan, #round(fbeta, ROUND_TO),
            'PR_AUC':     np.nan, #round(pr_auc, ROUND_TO),
            "RMSE":     round(reggres_metrics["RMSE"], round_to), 
            "MAE":      round(reggres_metrics["MAE"], round_to), 
            "R2":       round(reggres_metrics["R2"], round_to), 
            "Pearson":  round(reggres_metrics["Pearson"], round_to), 
            "Spearman": round(reggres_metrics["Spearman"], round_to)
        }

    else:
        ## PROCESS OTHER MODEL TYPE
        # Check predict_proba exists for a model
        try:
            y_scores = model.predict_proba(X)[:, 1]
        except (AttributeError, NotImplementedError):
            if hasattr(model, "decision_function"):
                y_scores = model.decision_function(X)
            else:
                y_scores = [0] * len(X)  # fallback for scoring

        # Normalize labels
        label_mapping = {'no': 0, 'yes': 1}

    
        # If y_pred is still strings (rare), map too
        if isinstance(y_pred[0], str):
            y_pred = pd.Series(y_pred).map(label_mapping)


        # Calculate metrics
        if pos_label is None: 
            accuracy = accuracy_score(y, y_pred)
            #pr_auc = average_precision_score(y, y_scores)
            pr_auc = np.nan
            precision = precision_score(y, y_pred, average='macro')
            recall = recall_score(y, y_pred, average='macro')
            fbeta = fbeta_score(y, y_pred, beta=1, average='macro')
        else:
            if y.dtype == 'object' or isinstance(y.iloc[0], str):
                y_same = y.map(label_mapping)
                pos_label_same = 1
            else:
                y_same = y
                pos_label_same = pos_label

            accuracy = accuracy_score(y_same, y_pred)
            pr_auc = average_precision_score(y_same, y_scores)
            precision = precision_score(y_same, y_pred, pos_label=pos_label_same, zero_division=0)
            recall = recall_score(y_same, y_pred, pos_label=pos_label_same, zero_division=0)
            fbeta = fbeta_score(y_same, y_pred, pos_label=pos_label_same, beta=2, zero_division=0)

        #pr_auc = average_precision_score(y_same, y_pred, pos_label=pos_label_same)

        # Prepare row for DataFrame
        row = {
            'Model': model_name,
            'Split': data_split,
            'Model_Type': model_type,
            'Time':       round(delta_time, round_to),
            'Accuracy':   round(accuracy, round_to),
            'Precision':  round(precision, round_to),
            'Recall':     round(recall, round_to),
            'FBeta':      round(fbeta, round_to),
            'PR_AUC':     round(pr_auc, round_to),
            "RMSE":     round(reggres_metrics["RMSE"], round_to), 
            "MAE":      round(reggres_metrics["MAE"], round_to), 
            "R2":       round(reggres_metrics["R2"], round_to), 
            "Pearson":  round(reggres_metrics["Pearson"], round_to), 
            "Spearman": round(reggres_metrics["Spearman"], round_to)
        }

 
        if plot_pr_auc:
            plot_title = model_name + '_' + data_split
            plot_pr_auc(plot_title, precision, recall, pr_auc)
    
    ## Add metric to df_model in expected order
    row_df = pd.DataFrame([row])[cols_df_model]  

    # ----- append or replace --------------------------------------
    mask = (df_models_eval['Model'] == model_name) & (df_models_eval['Split'] == data_split)
    
    if mask.any():
        df_models_eval.loc[mask, :] = row_df.values
    else:
        df_models_eval = pd.concat([df_models_eval, row_df], ignore_index=True)

    # # check row doesn't exist and replace if does
    # condition = (df_models_eval['Model'] == model_name) & (df_models_eval['Split'] == data_split)

    # if df_models_eval[condition].empty:
    #     print('if empty')
    #     df_models_eval = pd.concat([df_models_eval, pd.DataFrame([row])], ignore_index=True)
    # else:
    #     print('else')
    #     df_models_eval.loc[condition, :] = pd.DataFrame([row]).values

    ## ADD classfication metrics for regression models
    # if threshhold_per_drug is not None and model_type == 'regression':
    #     df_models_eval, X = add_binary_metrics(
    #         df_models_results=df_models_eval,
    #         model_name=model_name,
    #         split_name='Train',
    #         df=X,
    #         y_true=y,
    #         y_pred=y_pred,  #X[f"{model_name}_y_pred"], # THIS IS NOT ADDED TO X
    #         threshold_per_drug=threshhold_per_drug,
    #         pos_label=1,
    #         round_to=ROUND_TO
    #     )

    return df_models_eval



def get_model_condition(model_name=None, data_split=None, df_models_eval=None):
    '''
    get_model_condition(model_name, data_split):
    args:
        model_name (str): name of model in df_models_eval
        data_split (str):  Train or Test

    returns:
        conditions (dataframe indx): index of dataframe of data for model_name and data_split
    '''
    if df_models_eval is None:
        print("Provide the df_models_eval dataframe to print 'get_model_condition(df_model=df_model)")
        return
    elif data_split is None:
        condition = (df_models_eval['Model'] == model_name)
    elif model_name is None:
        condition = (df_models_eval['Split'] == data_split)
    else:
        condition = (df_models_eval['Model'] == model_name) & (df_models_eval['Split'] == data_split)
    
    return condition

# Mean, median, std of r2 per feature
# build a table of r2 by drug, cosmic_id, cell info and drug info
def plot_r2_by_feature(df, y_test, y_pred, floor=None, model_name=None):
    features = [ 'drug_id', 'target', 'pathway', 'cancer_type', 'tissue_desc_1',
            'tissue_desc_2','growth_properties']
    r2_mean_dict = {key: [] for key in features}
    r2_median_dict = {key: [] for key in features}
    for feature in features:
        r2_list = []
        #r2_array = np.empty(len(df_val[feature].unique()))
        for feat_name in df[feature].unique():
            idx = (df[feature] == feat_name)
            #feat_vals.append(feat_val)
            
            if idx.sum() > 1:  # Need at least 2 samples to compute R²
                r2 = r2_score(y_test[idx], y_pred[idx])
                if floor is not None:
                    r2 = floor if r2 < floor else r2
                r2_list.append(r2)
        r2_mean_dict[feature] = np.array(r2_list)
        #r2_median_dict[feature] = np.median(np.array(r2_list))

    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(r2_mean_dict)

    ax.set_ylabel("Mean R² (per feature)")
    if floor is not None:
        ax.set_ylim(floor, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(
        f"{model_name}:\n"
        "Mean R² values by drug or cell information (Validation Set)"
    )
    ax.set_ylabel("Mean R² (per feature)")
    return fig
    
def plot_scatter_relationship(x, y, xlabel="Actual LN_IC50", ylabel="Predicted LN_IC50",title=None):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=x, y=y, alpha=0.6)
    plt.plot([x.min(), y.max()], [x.min(), x.max()], 'r--', lw=2) # Diagonal line
    plt.xlabel("Actual LN_IC50")
    plt.ylabel("Predicted LN_IC50")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
