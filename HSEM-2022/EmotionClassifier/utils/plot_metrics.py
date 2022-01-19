"""
plot train loss, train acc, valid loss, valid acc
"""

import matplotlib.pyplot as plt
import os
from config import mark

f = open(os.path.join('..', 'log', f'train_{mark}.txt'), 'r')
data = f.readlines()[1:]
epochs = len(data)
print(epochs)
train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

for line in data:
    line = line.strip().split('\t')
    train_loss.append(line[0])
    train_acc.append(line[1])
    valid_loss.append(line[2])
    valid_acc.append(line[3])
train_loss = list(map(float, train_loss))
train_acc = list(map(float, train_acc))
valid_loss = list(map(float, valid_loss))
valid_acc = list(map(float, valid_acc))

plt.plot(range(epochs), train_loss, linewidth=2, color='b', label='train loss')  
plt.plot(range(epochs), valid_loss, linewidth=2, color='r', label='valid loss')  
plt.title("loss", fontsize=24, color='r')  
plt.xlabel("epochs", fontsize=14, color='g')  
plt.ylabel("loss", fontsize=14, color='g')
plt.legend()
plt.savefig("loss.png")
plt.show()

plt.plot(range(epochs), train_acc, linewidth=2, color='b', label='train acc')  
plt.plot(range(epochs), valid_acc, linewidth=2, color='r', label='valid acc')  
plt.title("loss", fontsize=24, color='r')  
plt.xlabel("epochs", fontsize=14, color='g')  
plt.ylabel("loss", fontsize=14, color='g')
plt.legend()
plt.savefig("acc.png")