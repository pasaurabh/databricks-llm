# Databricks notebook source
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # viz
import matplotlib.pyplot as plt # viz
from scipy import stats
import json
from typing import List, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn import metrics, linear_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

train_df.groupby(['sus', 'evil'])[['timestamp']].count()

# COMMAND ----------

test_df.groupby(['sus', 'evil'])[['timestamp']].count()

# COMMAND ----------

train_df.groupby(['sus'])[['timestamp']].count()

# COMMAND ----------

test_df.groupby(['sus'])[['timestamp']].count()

# COMMAND ----------

train_df.sus.value_counts().plot(kind='bar', title='Label Frequency for sus label in Train Dataset')

# COMMAND ----------

def dataset_to_corr_heatmap(dataframe, title, ax):
    corr = dataframe.corr()
    sns.heatmap(corr, ax = ax, annot=True, cmap="YlGnBu")
    ax.set_title(f'Correlation Plot for {title}')

# COMMAND ----------

datasets = [train_df, test_df, validation_df]

entropy_values = []
for dataset in datasets:
    dataset_entropy_values = []
    for col in dataset.columns:
        if col == 'timestamp':
            pass
        else:
            counts = dataset[col].value_counts()
            col_entropy = stats.entropy(counts)
            dataset_entropy_values.append(col_entropy)
            
    entropy_values.append(dataset_entropy_values)

plt.boxplot(entropy_values)
plt.title('Boxplot of Entropy Values')
plt.ylabel("entropy values")
plt.xticks([0,1,2,3],labels=['','train', 'test', 'validate'])
plt.show()

# COMMAND ----------

train = train_df[["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]]
train_labels = train_df['sus']

# COMMAND ----------

def process_args_dataframe(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Processes the `args` column within the dataset
    """
    
    processed_dataframes = []
    data = df[column_name].tolist()
    
    # Debug counter
    counter = 0
    
    for row in data:
        if row == '[]': # If there are no args
            pass
        else:
            try:
                ret = process_args_row(row)
                processed_dataframes.append(ret)
            except:
                print(f'Error Encounter: Row {counter} - {row}')

            counter+=1
        
    processed = pd.concat(processed_dataframes).reset_index(drop=True)
    processed.columns = processed.columns.str.lstrip()
    
    df = pd.concat([df, processed], axis=1)
    
    return df

def prepare_dataset(df: pd.DataFrame, process_args=False) -> pd.DataFrame:
    """
    Prepare the dataset by completing the standard feature engineering tasks
    """
    
    df["processId"] = train_df["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
    df["parentProcessId"] = train_df["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
    df["userId"] = pd.to_numeric(df["userId"])  # Convert "userId" column to numeric values
    df["userId"] = train_df["userId"].map(lambda x: 0 if x < 1000 else 1)  # Map to OS/not OS
    df["mountNamespace"] = train_df["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)  # Map to mount access to mnt/ (all non-OS users) /elsewhere
    df["eventId"] = train_df["eventId"]  # Keep eventId values (requires knowing max value)
    df["returnValue"] = pd.to_numeric(df["returnValue"], errors="coerce")  # Convert "returnValue" column to numeric values
    df["returnValue"] = train_df["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))  # Map to success/success with value/error
    
    if process_args is True:
        df = process_args_dataframe(df, 'args')
        
    features = df[["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]]
    labels = df['sus']
        
    return features, labels

# COMMAND ----------

train_no_args_feats, train_no_args_labels = prepare_dataset(train_df)

# COMMAND ----------

train_df_feats, train_df_labels = prepare_dataset(train_df)
test_df_feats, test_df_labels = prepare_dataset(test_df)
val_df_feats, val_df_labels = prepare_dataset(validation_df)

# COMMAND ----------

clf = IsolationForest(contamination=0.1, random_state=0).fit(train_df_feats)

# COMMAND ----------

def metric_printer(y_true, y_pred):
    y_true = np.array([int(label) if label.isdigit() else -1 for label in y_true], dtype=np.int64)
    y_true[y_true == 0] = 1

    metric_tuple = precision_recall_fscore_support(y_true, y_pred, average="weighted", pos_label=-1)
    print(f'Precision:\t{metric_tuple[0]}')
    print(f'Recall:\t\t{metric_tuple[1]:.3f}')
    print(f'F1-Score:\t{metric_tuple[2]:.3f}')

# COMMAND ----------

y_pred= clf.predict(val_df_feats)
y_probas = clf.score_samples(val_df_feats)
metric_printer(val_df_labels, y_pred)

# COMMAND ----------

y_pred= clf.predict(test_df_feats)
y_probas = clf.score_samples(test_df_feats)
metric_printer(test_df_labels, y_pred)
