
from re import X
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import pipeline
from Data_Prep import *
from Models import *
from Model_results import *
import pickle




#Importing data
train_data = pd.read_csv('/home/burny/Personal Projects/Machine Learning Projects/Projects/Churn Prediction/Customer-Churn-Prediction/train_dataset.csv')
train_data

train_data.shape 
train_data.dtypes

#Prepare data & clean data
x,y = Prepare_data(train_data)



#Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.15, random_state=42)


#DecisionTreeClassifier
model = Models(X_train, y_train, X_test)

dcstree = model.DecisionTree() #Obtain predicted values
dcstree_score = Model_score(y_test, dcstree)
dcstree_matrix = Confusion_Matrix(y_test, dcstree)


#RandomFOrest
RF = model.RandomForestTree()
RF_score = Model_score(y_test, RF)
RF_matrix = Confusion_Matrix(y_test, RF)


#Esemble Learning

models = list()

decision_tree = Pipeline([('m', DecisionTreeClassifier())])
models.append(('decision', decision_tree))

random_forest =  Pipeline([('m', RandomForestClassifier(n_estimators=20, random_state=42))])
models.append(('randomforest', random_forest))

svc = Pipeline([('m', SVC())])
models.append(('svc', svc))

knn = Pipeline([('m', KNeighborsClassifier())])
models.append(('knn', knn))

esemble = VotingClassifier(estimators=models, voting='hard')

#Train the esemble learning model
esemble.fit(x, y)



#Check prediction on test data
test_data = pd.read_csv('/home/burny/Personal Projects/Machine Learning Projects/Projects/Churn Prediction/Customer-Churn-Prediction/test_dataset.csv')


test_data = Prepare_testdata(test_data)
test_data = test_data.drop(columns = ['id'], axis = 1)
test_data



result = esemble.predict(test_data)
result

#Save the model











