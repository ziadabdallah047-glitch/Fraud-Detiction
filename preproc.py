import pandas as pd

def transform_features(df, scaler):
    df_transformed = df.copy()
    cols_to_scale = ['Time', 'Amount']
    df_transformed[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df_transformed

