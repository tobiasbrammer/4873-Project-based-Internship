def smape(actual, predicted):
    return 100 / len(actual) * np.sum(np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))