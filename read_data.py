import numpy as np

converters = {83: lambda s: s.decode("utf-8")}
data = np.loadtxt("data_train.txt",dtype=str,delimiter=',',skiprows=1, usecols=range(1,89), converters=converters)

print(data)



