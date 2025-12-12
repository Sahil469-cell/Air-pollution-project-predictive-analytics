import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\Owner\Desktop\project dataset.csv") \
      .pivot_table(index=['city','station'], columns='pollutant_id', values='pollutant_avg') \
      .reset_index()[['PM2.5','PM10','NO2','SO2','CO','OZONE','NH3']].dropna()

y = df['PM2.5']
X1 = df[['PM10']]
X2 = df.drop('PM2.5', axis=1)

print("Simple R² =", round(LinearRegression().fit(X1,y).score(X1,y), 4))
print("Multi  R² =", round(LinearRegression().fit(X2,y).score(X2,y), 4))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(df.PM10, y, alpha=0.7)
plt.plot(df.PM10, LinearRegression().fit(X1,y).predict(X1))
plt.title('PM10 vs PM2.5')
plt.xlabel('PM10'); plt.ylabel('PM2.5')

plt.subplot(1,2,2)
p = LinearRegression().fit(X2,y).predict(X2)
plt.scatter(y, p, color='green', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.title('Predicted vs Actual')
plt.xlabel('Actual'); plt.ylabel('Predicted')

plt.suptitle('Simple vs Multiple Regression', fontsize=14)
plt.tight_layout()
plt.show()