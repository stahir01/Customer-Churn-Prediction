from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn import neighbors
from Model_results import *



class Models:
    def __init__(self, X_train, y_train, X_test) -> int:

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test

    def RandomForestTree(self):
        RF = RandomForestClassifier(n_estimators=20, random_state=42)
        RF = RF.fit(self.X_train, self.y_train)
        y_predict = RF.predict(self.X_test)

        return y_predict
    
    def SVM(self):
        clf = SVC(kernel='rbf')
        clf = clf.fit(self.X_train, self.y_train)
        clf_predict = clf.predict(self.X)
        
        return clf_predict

    def DecisionTree(self):
        decision_tree = DecisionTreeClassifier()
        decision_tree = decision_tree.fit(self.X_train, self.y_train)
        dtc_predict = decision_tree.predict(self.X_test)
        
        return dtc_predict

    def KNN(self):
        knn = KNeighborsClassifier(n_neighbors=6)
        knn.fit(self.X_train,self.y_train)
        knn_predict = knn.predict(self.X_test)
        
        return knn_predict
