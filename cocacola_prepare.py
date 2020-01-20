import pandas as pd

data = pd.read_csv("Cocacola_stock.csv", usecols=["Close/Last"])
data = data.values
file = open("Cocacola_prepared.csv", "w")

for i in range(20):
	file.write(f'f{i+1},')
file.write("target\n")

for i in range(20, len(data)-1):
	for j in range(i-20, i):
		file.write(str(data[j][0]) + ",")
	file.write(str(data[i][0]) + "\n")

file.close()