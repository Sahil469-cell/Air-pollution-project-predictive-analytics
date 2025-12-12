import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

df = pd.read_csv(r"C:\Users\Owner\Desktop\project dataset.csv")
df["pollutant_avg"] = pd.to_numeric(df["pollutant_avg"], errors="coerce")

pm25_avg = df[df.pollutant_id=="PM2.5"].groupby("city").pollutant_avg.mean()
pm10_avg = df[df.pollutant_id=="PM10"].groupby("city").pollutant_avg.mean()

city_data = pd.concat([pm25_avg, pm10_avg], axis=1).reset_index()
city_data.columns = ["city","PM25","PM10"]
city_data = city_data.dropna()
city_data["unsafe"] = (city_data["PM25"]>100).astype(int)

X = city_data[["PM25","PM10"]].values
y = city_data["unsafe"].values

svm_model = SVC(kernel="linear")
svm_model.fit(X, y)

x_range = np.linspace(X[:,0].min()-5, X[:,0].max()+5, 300)
y_range = np.linspace(X[:,1].min()-5, X[:,1].max()+5, 300)
xx, yy = np.meshgrid(x_range, y_range)
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10,7))
plt.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
plt.scatter(X[:,0], X[:,1], c=y, s=100, cmap="coolwarm", edgecolors="k")
plt.xlabel("PM2.5")
plt.ylabel("PM10")
plt.title("SVM Classification: Safe vs Unsafe Cities")
plt.grid(True)
plt.show()
