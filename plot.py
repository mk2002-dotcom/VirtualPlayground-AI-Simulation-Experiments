# Make one graph from some data files
import numpy as np
import matplotlib.pyplot as plt




data_num = 3
# If you specify each the color
color = ['red', 'blue', 'green']

fig, ax = plt.subplots()
for i in range(data_num):
	data = np.loadtxt(f'data{i}.txt')
	x = data[:,0]
	y = data[:,1]
	error = data[:,2]
 
	# If the data does not contain errors
    # ax.plot(x, y, marker='.', linestyle='-', label=f'data{i}')
	ax.errorbar(x, y, error, marker = '.', linestyle = '', label = f'data{i}', color = color[i])

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title('Data')
	#ax.set_xlim(0, 1)
	#ax.set_ylim(0, 1)
	ax.legend()
	fig.savefig("plot.png")