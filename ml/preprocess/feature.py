from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class Encoder:
    
    def __init__(self, cols):
        self._cols = cols
        self._le = LabelEncoder()
    
    def fit_transform(self, df):
        if isinstance(self._cols, str):
            col = self._cols
            df[col] = self._le.fit_transform(df[col])
            return df
        elif isinstance(self._cols, list):
            cols = self._cols
            for c in cols:
                df[c] = self._le.fit_transform(df[c])
            return df


class Scalers:
    
    def __init__(self):
        self._sc = StandardScaler()
    
    def fit_transform(self, x_train, x_test):
        x_train, x_test = self._sc.fit_transform(x_train), self._sc.transform(x_test)
        return x_train, x_test
