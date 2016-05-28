from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

input_file = "/home/deepita/Kaggle/Titanic/train_age.csv"
input_test_file = "/home/deepita/Kaggle/Titanic/test_age.csv"
df = pd.read_csv(input_file,header=0)
df_test = pd.read_csv(input_test_file,header=0)

feature_cols = ['Pclass','Sex','SibSp','Parch']
X = df[feature_cols]
y = df.Age
lm = LinearRegression()
lm.fit(X,y)

#print lm.coef_

Age = lm.predict(df_test)
Age = [np.median(Age) if i < 0 else i for i in Age]

print Age
np.savetxt('age_test_1.csv',Age)
print lm.score(X, y)