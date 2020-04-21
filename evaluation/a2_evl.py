# Third party Link: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
import numpy as np
import random
from sklearn.metrics import average_precision_score

# Random input of true binary labels, size 10
y_true = np.random.randint(2, size = 10)

# Random target scores, size 10
y_scores = np.random.random((10,1))
print(y_true)
print(y_scores)
print(average_precision_score(y_true, y_scores))
