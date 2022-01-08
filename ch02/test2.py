
import pandas as pd

from sklearn import linear_model
from matplotlib import pyplot as plt

df = pd.read_csv("./ch02/whdata.csv")

print(df)


height = df["Height"] * 2.54
weight = df["Weight"] * 0.453

height = height.to_numpy().reshape(-1,1)



lm = linear_model.LinearRegression()

lm.fit(height, weight)

print("a")
print(lm.coef_)
print("b")
print(lm.intercept_)

print( lm.score(height, weight) )


print( lm.predict([[200]]) )





plt.plot(height, weight, "bo")
plt.plot( [130, 210] , [lm.predict([[130]]), lm.predict([[210]])], 'r')
plt.show()
