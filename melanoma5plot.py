#imports

import melanoma5
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

#print confusion matrix
#work in progress, dont really need, only really need history plot saved

#disp = ConfusionMatrixDisplay(confusion_matrix = melanoma5.cm_data)
#disp.plot()
#plt.show()

#from csv
#conf_matrix = pd.read_csv("confusion_matrix_data.csv")


#print val

melanoma5.history_data.plot(figsize=(8,5))
plt.ylim(0, 1)
plt.show()