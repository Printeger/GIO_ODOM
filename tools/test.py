import numpy as np

num_ = 100
test_gyro = np.zeros(num_)

for i in range(num_):
    test_gyro[i] = i - 50
    i = i + 1

first_flag = True
last_gyro = test_gyro[0]
mean_gyro = test_gyro[0]
N = 1

if first_flag:
    last_gyro = test_gyro[0]
    mean_gyro = test_gyro[0]
    N = 1

for i in range(num_):
    curr_gyro = test_gyro[i]
    mean_gyro = mean_gyro + (curr_gyro - mean_gyro) / N
    N = N + 1

    print("No. ", i, "mean_gyro: ", mean_gyro)
