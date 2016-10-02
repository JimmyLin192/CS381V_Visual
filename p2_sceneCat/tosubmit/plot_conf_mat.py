import itertools
import numpy as np
import matplotlib.pyplot as plt
import plot_confusion_matrix

# confusion matrix
cnf_matrix = [[0 for x in range(25)] for y in range(25)]
f = open('./no_pretrain_result')
for line in f:
    info = line.split(',')
    i, j, cnt = int(info[0]), int(info[1]), int(info[2])
    cnf_matrix[i][j] = cnt
f.close()

# all classnames
f = open('classnames')
classes = []
for line in f:
    classes.append(line)
f.close()

plt.figure()
plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=90)
plt.yticks(tick_marks, classes)
plt.ylabel('Real label')
plt.xlabel('Predicted label')
plt.colorbar()
plt.tight_layout()
plt.show()
