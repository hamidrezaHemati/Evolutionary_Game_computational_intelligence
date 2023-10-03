import numpy as np
import matplotlib.pyplot as plt

def readFile():
    data = []
    with open('fitnesses.txt', 'r') as file:
        input = file.readlines()
        for d in input:
            data.append(d.replace('\n', ''))
        file.close()
    tokenize_data = []
    for i in range(len(data)):
        tokens = data[i].split()
        tokens = [float(token) for token in tokens]
        tokenize_data.append(tokens)
    return tokenize_data

def main():
    data = readFile()
    min = []
    max = []
    average = []
    for d in data:
        min.append(d[0])
        average.append(d[1])
        max.append(d[2])
    min = np.array(min)
    max = np.array(max)
    average = np.array(average)
    generations = len(data)
    print(type(generations))
    generations = np.arange(0, generations, step=1)
    # plt.plot(generations, min, 'r--', generations, average, 'bs', generations, max, 'g^')
    plt.plot(generations, min, label='min')
    plt.plot(generations, average, label='average')
    plt.plot(generations, max, label='max')
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.legend()
    plt.show()

main()