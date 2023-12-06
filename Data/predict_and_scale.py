def predict_and_scale(df, dfScaled, model, sName, lModelVars, lJobNo, bConst=False, iBatchSize=1):
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
    from statsmodels.regression.linear_model import RegressionResultsWrapper
    from keras.models import Sequential
    import multiprocessing
    import pandas as pd
    import numpy as np

    # If bConst is True, then add 'intercept' to the list of model variables
    if bConst:
        lModelVars = lModelVars + ['intercept']
    else:
        pass

    # x_scaler = joblib.load("./.AUX/x_scaler.save")
    y_scaler = joblib.load("./.AUX/y_scaler.save")

    predicted_col = f'predicted_{sName}'

    for iJobNo in lJobNo:
        lIndex = df[df['job_no'] == iJobNo].index.tolist()
        if len(lIndex) > 1:
            for i in range(len(lIndex)):
                lIndexSeq = lIndex[0:i + 1]
                # Predict using the provided model
                if isinstance(model, RegressionResultsWrapper):
                    dfScaled.loc[lIndex[i], predicted_col] = model.predict(dfScaled.loc[lIndexSeq, lModelVars]).values[i]
                elif isinstance(model, Sequential):
                    dfScaled.loc[lIndex[i], predicted_col] = model.predict(dfScaled.loc[lIndexSeq, lModelVars],
                                                                           batch_size=iBatchSize,
                                                                           use_multiprocessing=True,
                                                                           workers=multiprocessing.cpu_count(),
                                                                           verbose=0
                                                                           )[i][0]
                else:
                    dfScaled.loc[lIndex[i], predicted_col] = model.predict(dfScaled.loc[lIndexSeq, lModelVars])[i]
                # Rescale and update the original dataframe
                df.loc[lIndex[i], predicted_col] = y_scaler.inverse_transform(
                    dfScaled.loc[lIndex[i], predicted_col].reshape(-1, 1)).reshape(-1)
        else:
            pass
