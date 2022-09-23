import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, ConfusionMatrixDisplay




def Model_score(y_test, y_pred):
  #Precision, accuracy, f1_score
  print("Accuracy:",accuracy_score(y_test, y_pred))
  print("Precision: ", precision_score(y_test, y_pred))
  print("f1_score: ", f1_score(y_test, y_pred))
  print(classification_report(y_test, y_pred))

def Confusion_Matrix(y_test, y_pred):
  cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
  disp.plot()
  plt.show()
