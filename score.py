import torch
import matplotlib.pyplot as plt

right = 100
n = 1
f = open('./test.txt','a')
f.write("正确率：{}".format(str(right/n)))
f.close()

