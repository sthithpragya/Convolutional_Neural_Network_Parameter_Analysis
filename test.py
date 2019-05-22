from matplotlib import pyplot as plt
import numpy as np


names = ['layer1', 'layer2', 'layer3']
score_loss = [1, 2, 3]
score_acc = [1.4, 2.6, 4]

N = 3
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

rects1 = ax.bar(ind, score_loss, width, color='r')
rects2 = ax.bar(ind+width, score_acc, width, color='g')

ax.set_ylabel('Scores')
ax.set_xticks(ind+width)
ax.set_xticklabels( (names[0], names[1], names[2]) )
ax.legend( (rects1[0], rects2[0]), ('loss', 'acc') )

plt.show()
