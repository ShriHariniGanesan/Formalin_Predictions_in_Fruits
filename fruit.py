import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm

import tkinter as tk
#1-apple 3-orange
data = pd.read_csv('C:/HARI/Harini/Final yr proj/fruit_data.csv')

sns.countplot(x="State", data=data, palette="bwr")
plt.xlabel("State (0=UnSafe , 1=Safe)")
plt.show()

X = data.iloc[:,0:2]
y = data.iloc[:,-1]

#Create a svm Classifier
clf = svm.SVC(kernel='rbf')
clf.fit(X, y)
#setting the train and test data size
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.30,random_state=6)
clf.fit(X_train,y_train)
y_test_pred=clf.predict(X_test)
prediction_output=pd.DataFrame(data=[y_test_pred,y_test.values])
print("Accuracy of SVM: {:.1f}%\n\n".format(accuracy_score(y_test, y_test_pred)*100))
svmconf=confusion_matrix(y_test,y_test_pred)
print("Confusion matrix for SVM:\n",svmconf)
#Create a KNN Clasifier
knn = KNeighborsClassifier(n_neighbors = 32)
knn.fit(X,y)
y_pred=knn.predict(X)
y_predict=pd.DataFrame(data=[y_pred,y.values])
#setting the train and test data size
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,random_state=6,shuffle=True)
knn.fit(X_train,y_train)
y_test_pred=knn.predict(X_test)
prediction_output=pd.DataFrame(data=[y_test_pred,y_test.values])
# print(prediction_output.transpose)
clas_rep = classification_report(y_test, y_test_pred)
print("Classification Report:",)
print (clas_rep)
print("\n\naccuracy of KNN :{:.1f}%".format(accuracy_score(y_test,y_test_pred)*100))
cm = confusion_matrix(y_test, y_test_pred)

print("Confusion Matrix:",)
print(cm)

def predict():
    a=e1.get()
    b=e2.get()
    res=knn.predict([[a,b]])
    if res==1:
        myText.set("Safe")
    else:
        myText.set("Unsafe")
#Tkinter to test prediction of knn
master = tk.Tk()
myText=tk.StringVar()
master.title("Predictive Model")
tk.Label(master, text="Fruit_Label").grid(row=0, sticky=tk.W)
tk.Label(master, text="ppm").grid(row=1, sticky=tk.W)
tk.Label(master, text="Prediction:").grid(row=3, sticky=tk.W)
result=tk.Label(master, text="", textvariable=myText).grid(row=3,column=1, sticky=tk.W)
 
e1 = tk.Entry(master)
e2 = tk.Entry(master)
 
e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
 
b = tk.Button(master, text="Calculate", command=predict)
b.grid(row=0, column=2,columnspan=2, rowspan=2,sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)
 
 
tk.mainloop()
