from sklearn import datasets
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
boston=datasets.load_boston()
# print(boston)
X=boston.data
Y=boston.target
# print(X)
# print(X.shape)
import pandas as pd
pd.set_option('display.max_rows',600)
pd.set_option('display.max_columns',14)

df=pd.DataFrame(X)
df.columns=boston.feature_names
df.describe()
print(df.describe().to_string())

