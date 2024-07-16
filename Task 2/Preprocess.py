import numpy as np
from sklearn.utils import shuffle


def preprocess_data(df, selected_features, selected_classes,Activation="sigmoid"):
    # Filter data based on selected classes
    df_filtered = df[df['Class'].isin(selected_classes)]

    # Handle null values
    for feature in selected_features:
        df_filtered.loc[:, feature] = df_filtered[feature].fillna(df_filtered[feature].mean())
    
    # Encoding and dropping columns
    # if Activation=="sigmoid":
    #     df_filtered['Class_encoded'] = np.where(df_filtered['Class'] == selected_classes[0], 1,0)
    # else:
    #     df_filtered['Class_encoded'] = np.where(df_filtered['Class'] == selected_classes[0], 1, -1)

    # mapping = {'BOMBAY': -1, 'CALI': 0, 'SIRA': 1}
    # df_filtered.loc[:, 'Class_encoded'] = df_filtered['Class'].map(mapping)
    # df_filtered.drop(columns=['Class'], inplace=True)
    
    # Outlier handling
    def find_range(col, df, class_value):
        class_rows = df[df['Class'] == class_value][col]
        Q1, Q3 = class_rows.quantile(0.25), class_rows.quantile(0.75)
        IQR = Q3 - Q1
        lower_range = Q1 - 1.5 * IQR
        upper_range = Q3 + 1.5 * IQR
        return lower_range, upper_range
    
    for col in selected_features:
        for class_value in selected_classes:
            lower_limit, upper_limit = find_range(col, df_filtered, class_value)
            class_rows = df_filtered['Class'] == class_value
            df_filtered.loc[class_rows, col] = df_filtered.loc[class_rows, col].clip(lower_limit, upper_limit)
    
    # Scaling Standard Scalling
    df_filtered.loc[:, selected_features] = (df_filtered[selected_features] - df_filtered[selected_features].mean()) / df_filtered[selected_features].std()
    

    # Splitting
    X = df_filtered[selected_features]
    y = df_filtered["Class"]
    
    return X, y