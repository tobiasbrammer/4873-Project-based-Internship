def predict_and_scale(df, dfScaled, model, sName, lModelVars, lJobNo):
    """
    Function to predict and scale data for each job number.

    Parameters:
    df (DataFrame): The original DataFrame.
    dfScaled (DataFrame): The scaled DataFrame.
    model: The predictive model to be used.
    name (str): The name to be used for creating dynamic column names.
    lJobNo (list): List of job numbers to process.

    Returns:
    None: The function modifies the df and dfScaled DataFrames in place.
    """
    import joblib
    import pandas as pd
    import numpy as np

    x_scaler = joblib.load("./.AUX/x_scaler.save")
    y_scaler = joblib.load("./.AUX/y_scaler.save")

    predicted_col = f'predicted_{sName}'

    for iJobNo in lJobNo:
        lIndex = dfScaled[dfScaled['job_no'] == iJobNo].index.tolist()
        if len(lIndex) > 1:
            for i in range(len(lIndex)):
                lIndexSeq = lIndex[0:i + 1]
                # Predict using the provided model
                dfScaled.loc[lIndex[i], predicted_col] = model.predict(dfScaled.loc[lIndexSeq, lModelVars])[i]
                # Rescale and update the original dataframe
                df.loc[lIndex[i], predicted_col] = y_scaler.inverse_transform(
                    dfScaled.loc[lIndex[i], predicted_col].reshape(-1, 1)).reshape(-1)
        else:
            pass
