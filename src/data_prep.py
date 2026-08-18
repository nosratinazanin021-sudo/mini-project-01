import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD DATA
# ==========================================

def load_data(file_path):

    return pd.read_csv(file_path)

# ==========================================
# 2. BASIC DATA OVERVIEW
# ==========================================

def get_basic_info(df):

    return {
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 1,  
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict()
    }

def preview_data(df):

    return df.head()

def get_statistics(df):

    return df.describe()

# ==========================================
# 3. DATA QUALITY CHECKS
# ==========================================

def amount_check(df):
    print(f"\n checking amount { df['Amount'].describe()}")

    return df['Amount'].describe()

def time_check(df):
    print(f"\n checking amount { df['Time'].describe()}")

    return df['Time'].describe()

def check_missing_values(df):

    return df.isnull().sum()

def check_duplicates(df):

    return df.duplicated().sum()

def remove_duplicates(df):
    df_clean=df.drop_duplicates()
    print(f"new samples:{df_clean.shape[0]}")

    return df_clean

# ==========================================
# 4. TARGET DISTRIBUTION
# ==========================================

def get_class_distribution(df, target_col='Class'):
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True)

    return {
        'count': counts,
        'percentage': percentages
    }

# ==========================================
# 5. COMPREHENSIVE REPORT 
# ==========================================
def data_preparation_report (df, target_col='Class'):
     
    basic = get_basic_info(df)
    print(f"\nnumber of samples:{basic['n_samples']}")
    print(f"\nnumber of features:{basic['n_features']}")
    print(f"\nfirst five rows of dataset:{preview_data(df)}")
    print(f"\ndescriptive statistic:{get_statistics(df)}")

    
    print(f"\nmising values:{check_missing_values(df)} ")

    dup_count = check_duplicates(df)
    print(f"\nduplicate rows:{dup_count}")
    dist = get_class_distribution(df, target_col)
    print(f"\n Class distribution (count):\n{dist['count']}")
    print(f"\n Class distribution (percent):\n{dist['percentage']}")

    return {

        'basic_info': basic,
        'preview': preview_data(df),
        'statistics': get_statistics(df),
        'missing_values': check_missing_values(df),
        'duplicates': dup_count,
        'class_distribution': dist
    }

# ==========================================
# 6. SPLIT DATA 
# ==========================================

def split_data(df , target_col='Class'):
    X=df.drop(columns=[target_col])
    y=df[target_col]

    X_train , X_test , y_train , y_test = train_test_split(
        X ,
        y ,
        test_size=0.2 ,
        stratify=y ,
        random_state=42
    )
    print(f"\nx_train:{X_train.shape}")
    print(f"x_test:{X_test.shape}")
    print(f"\ny_train:{y_train.value_counts(normalize=True)}")
    print(f"y_test:{y_test.value_counts(normalize=True)}")

    return X_train , X_test , y_train , y_test 


#==================================
# calling function :
#==================================

if __name__ == "__main__":
    df = load_data('data/creditcard.csv')
    
    
    report = data_preparation_report(df, target_col='Class')
    
    X_train, X_test, y_train, y_test = split_data(df, target_col='Class')
   