# Make a histogram with Gaussian
import numpy as np
import matplotlib.pyplot as plt


min, max = 0, 1
data = np.random.uniform(min, max, 50)
# If you have a data file
# data = np.loadtxt("data.txt")
# min, max = np.min(data), np.max(data)

bin = 30
width = 0.5 * (max - min) / bin
histo = [0] * bin
bin_val = [0] * bin
for k in range(bin):
    bin_val[k] = min + (k + 0.5) * (max - min) / bin
    for i in range(len(data)):
        histo[k] += np.exp(- ((data[i] - bin_val[k]) / width)**2)
        
    histo[k] = histo[k] / float(len(data))
    
    
fig, ax = plt.subplots()
ax.bar(bin_val, histo)
fig.savefig("Histogarm.png")