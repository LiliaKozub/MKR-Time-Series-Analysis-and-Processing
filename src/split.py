def train_val_test_split(ts, val_size=24, test_size=12):
    n = len(ts)
    if val_size + test_size >= n:
        raise ValueError("Занадто великий розмір val+test для довжини серії.")
    train_end = n - val_size - test_size
    train = ts.iloc[:train_end]
    val = ts.iloc[train_end:train_end+val_size]
    test = ts.iloc[train_end+val_size:]
    return train, val, test
