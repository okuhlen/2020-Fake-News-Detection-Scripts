import numpy as np

array = np.zeros(10)
print(array)
newArray = array.copy()
newArray = newArray + 100
print(newArray)
print(array)
print(newArray[1:4])

