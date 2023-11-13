# ODD2023-Datascience-Ex06
EX-06 FEATURE TRANSFORMATION
### AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

### EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

### ALGORITHM:
Step1: Read the given Data.

Step2: Clean the Data Set using Data Cleaning Process.

Step3: Apply Feature Transformation techniques to all the features of the data set.

Step4: Print the transformed features.

### PROGRAM:
```
Developed by : Niraunjana Gayathri G R
Register No : 212222230096
```
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df

df.skew()

df.head()

df.isnull().sum()

df.info()
df.describe()

np.log(df["Highly Positive Skew"])
np.reciprocal(df["Highly Positive Skew"])
np.sqrt(df["Highly Positive Skew"])

np.square(df["Highly Positive Skew"])
df["Highly Positive Skew_boxcox"],parameter=stats.boxcox(df["Highly Positive Skew"])
df
df["Moderate Negative Skew_yeojohnson"],parameter=stats.yeojohnson(df["Moderate Negative Skew"])
df

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

df1=df.copy()
sm.qqplot(df["Moderate Negative Skew"],fit=True,line='45')
plt.show()

sm.qqplot(df["Moderate Negative Skew_1"],line='45')
plt.show()

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],fit=True,line='45')
plt.show()

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```

### OUTPUT
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex06/assets/119395610/b0f6a4e7-5a95-4b94-917c-d97001242ecf)
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex06/assets/119395610/1f00a6f4-dbf8-4afd-8466-6811ff050665)
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex06/assets/119395610/0ba0ce3e-f6c9-4957-a37f-75a8dca4365f)
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex06/assets/119395610/e7d8692a-6662-44ff-8ae2-0827d12d6fc8)
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex06/assets/119395610/8d4c5f58-056a-4511-9359-6582cfd30d74)
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex06/assets/119395610/b9f725d3-96aa-4d24-b5a9-fdefbe1ed139)
![image](https://github.com/niraunjana/ODD2023-Datascience-Ex06/assets/119395610/9fbb2a52-b803-4a5f-922d-9d856b123136)
