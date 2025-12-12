import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r"C:\Users\Owner\Desktop\project dataset.csv")

city_avg = df[df.pollutant_id=="PM2.5"].groupby("city").pollutant_avg.mean().reset_index()
city_avg.columns = ["city", "PM2.5"]

city_avg["High"] = city_avg["PM2.5"] > 100

city_avg = city_avg.dropna()

X = city_avg[["PM2.5"]]
y = city_avg["High"]

model = LogisticRegression().fit(X, y)

safe_cities = len(city_avg[city_avg.High == 0])
unsafe_cities = len(city_avg[city_avg.High == 1])

print("\nSafe Cities:", safe_cities)
print("Unsafe Cities:", unsafe_cities)

plt.figure(figsize=(7,5))
plt.bar(["Safe Cities", "Unsafe Cities"], [safe_cities, unsafe_cities])
plt.title("Number of Safe vs Unsafe Cities")
plt.ylabel("Count")
plt.grid(alpha=0.3)
plt.show()
