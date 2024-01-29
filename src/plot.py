import matplotlib.pyplot as plt
import sys

file = open(sys.argv[1], "r")

lines = file.readlines()

x = []
y = []

for line in lines:
    line = line.split("\t")
    x.append(float(line[0]))
    y.append(float(line[1]))

plt.plot(x,y)
plt.show()