import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv('data/creditcard.csv')

def data_preparation_report (df, target_col='Class'):
    print(f"\nnumber of samples:{df.shape[0]}")
    print(f"\nnumber of features:{df.shape[1]-1}")
    print("\ncolumns info:")
    df.info()
    print(f"\nfirst five rows of dataset:{df.head()}")
    print(f"\ndescriptive statistic:{df.describe()}")
    distribution_count=df[target_col].value_counts()
    print(f"\nclass distribution(count): {distribution_count}")
    distribution_percent=df[target_col].value_counts(normalize=True)
    print(f"\nclass distribution(%):{distribution_percent}")
    missing = df.isnull().sum()
    print(f"\nmising values:{missing} ")
    dup_count=df.duplicated().sum()
    print(f"\nduplicate rows:{dup_count}")
    return {
        'n_samples':df.shape[0] ,
        'n_feature':df.shape[1]-1,
        'missing_values':missing,
        'duplicated':dup_count,
        'class_distribution(count)':distribution_count,
        'class distribution(%)':distribution_percent
    }


def remove_duplicates(df):
    df_clean=df.drop_duplicates(keep='first')
    print(f"new samples:{df_clean.shape[0]}")
    return df_clean

def split_data(df , target_col='Class'):
    X=df.drop(columns=[target_col])
    y=df[target_col]
    X_train , X_test , y_train , y_test = train_test_split(
        X ,y , test_size=0.2 , stratify=y , random_state=42
    )
    print(f"\nx_train:{X_train.shape}")
    print(f"x_test:{X_test.shape}")
    print(f"\ny_train:{y_train.value_counts(normalize=True)}")
    print(f"y_test:{y_test.value_counts(normalize=True)}")
    return X_train , X_test , y_train , y_test 

def scaling_data(X_train , X_test , cols_to_scale=['Time','Amount']):
    scaler=StandardScaler()
    X_train_scaled=X_train.copy()
    X_test_scaled=X_test.copy()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    print(f"\nx_train_scaled:{X_train_scaled[cols_to_scale].head()}")
    print(f"\nx_test_scaled:{X_test_scaled[cols_to_scale].head()}")
    return X_train_scaled ,X_test_scaled, scaler



report=data_preparation_report(df)
df=remove_duplicates(df) 
X_train , X_test , y_train , y_test=split_data(df)
X_train_scaled ,X_test_scaled, scaler=scaling_data(X_test , X_test)