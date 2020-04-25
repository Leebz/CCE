import numpy as np
import matplotlib.pyplot as plt


def plotLearning(scores, filename, x=None, window=5):
    print("------------------------Saving Picture--------------------------------")
    N = len(scores)
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])

    if x is None:
        x = [i for i in range(N)]

    plt.ylabel('Score')
    plt.xlabel('Game')

    plt.plot(x, running_avg, "r", label="our")
    plt.legend()

    plt.savefig(filename)
    plt.close()


def read_txt(path):
    res = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            line = list(map(int, line)) # 转为整型数组
            res.append(line)
    res = np.concatenate(res, axis=None)
    return res

def file_name_generator(seed, init_capacity):
    pass


if __name__ == '__main__':
    data = read_txt("../Dataset/data1.txt")
    print(len(data))

