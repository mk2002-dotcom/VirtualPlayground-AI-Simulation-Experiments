# Calculate jack knife error
import numpy as np


# Any function
def func(x):
    return sum(x)


def jack_knife(data, bin_size):
    sample_num = int(len(data) / bin_size)
    data_jack = [0] * (len(data) - bin_size)
    sample = [0] * sample_num

    for i in range(sample_num):
        for j in range(len(data)):
            k = 0
            if j < i * bin_size or (i + 1) * bin_size <= j:
                data_jack[k] += data[j]
                k += 1
        sample[i] = func(data_jack)

    sample_mean = np.mean(sample)
    var = 0
    for i in range(sample_num):
        var += (sample[i] - sample_mean) ** 2
    error = np.sqrt(var * (sample_num - 1) / sample_num)
    return error


# If you have a data file
# data = np.loadtxt('data.txt')
data = [1, 2, 3, 4, 5, 6]
# len(data) / bin_size must be an integer
bin_size = 2

error = jack_knife(data, bin_size)
print(f"Jack knife error:{error}")
