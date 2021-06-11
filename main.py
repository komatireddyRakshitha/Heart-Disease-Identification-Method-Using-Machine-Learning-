from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score,precision_score,recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense

main = tkinter.Tk()
main.title("Heart Disease")
main.geometry("1300x1200")
# Setting up the column
column = ['sbp','tobacco','ldl','adiposity','famhist','type','obesity','alcohol','age','chd']

def upload():
    global filename
    global data
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = ".")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")

def importdata():
    global filename
    global df
    df = pd.read_csv(filename)
    df.columns=column
    text.insert(END,"Data Information:\n"+str(df.head())+"\n")
    text.insert(END,"Columns Information:\n"+str(df.columns)+"\n")

def preprocess():
    global df
    # Feature Scaling, making categorical data precise 
    
    encoder = LabelEncoder()
    df['famhist']=encoder.fit_transform(df['famhist'])
    df['chd']=encoder.fit_transform(df['chd'])
    scale = MinMaxScaler(feature_range =(0,100))
    text.insert(END,"Preprocess Done\n")
    #setting scale of max min value for sbp in range of 0-100, normalise
    df['sbp'] = scale.fit_transform(df['sbp'].values.reshape(-1,1))
    df.head(50).plot(kind='area',figsize=(10,5))
    plt.figure(0)
    df.plot(x='age',y='obesity',kind='scatter',figsize =(10,5))
    plt.figure(1)
    df.plot(x='age',y='tobacco',kind='scatter',figsize =(10,5))
    plt.figure(2)
    df.plot(x='age',y='alcohol',kind='scatter',figsize =(10,5))
    plt.figure(3)
    df.plot(kind = 'hist',figsize =(10,5))
    plt.figure(4)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange',medians='DarkBlue', caps='Gray')
    df.plot(kind='box',figsize=(10,6),color=color,ylim=[-10,90])
    plt.show()
    

def ttmodel():
    global df
    global X_train, X_test, y_train, y_test
    # splitting the data into test and train  having a test size of 20% and 80% train size
    from sklearn.model_selection import train_test_split
    col = ['sbp','tobacco','ldl','adiposity','famhist','type','obesity','alcohol','age']
    X_train, X_test, y_train, y_test = train_test_split(df[col], df['chd'], test_size=0.2, random_state=1234)
    text.insert(END,"Shape of Train Data: "+str(X_train.shape)+"\n")
    text.insert(END,"Shape of Test Data: "+str(X_test.shape)+"\n")
    sns.set()
    sns.heatmap(X_train.head(10),robust = True)
    plt.show()

def models():
    global X_train, X_test, y_train, y_test
    global svm_result,knn_result,ann_result
    global recall_svm,recall_knn,recall_ann
    global precision_svm,precision_knn,precision_ann
    svm_clf = svm.SVC(kernel ='linear')
    svm_clf.fit(X_train,y_train)
    y_pred_svm =svm_clf.predict(X_test)
    svm_result = accuracy_score(y_test,y_pred_svm)
    print("Accuracy :",svm_result)
    recall_svm = recall_score(y_test,y_pred_svm)
    precision_svm = precision_score(y_test,y_pred_svm)
    text.insert(END,"Accuracy of SVM: "+str(svm_result)+"\n")
    text.insert(END,"Recall of SVM: "+str(recall_svm)+"\n")
    text.insert(END,"Precision of SVM: "+str(precision_svm)+"\n")

    knn_clf = KNeighborsClassifier(n_neighbors =5,n_jobs = -1,leaf_size = 60,algorithm='brute')
    knn_clf.fit(X_train,y_train)
    y_pred_knn = knn_clf.predict(X_test)
    knn_result = accuracy_score(y_test,y_pred_knn)
    recall_knn = recall_score(y_test,y_pred_knn)
    precision_knn = precision_score(y_test,y_pred_knn)
    text.insert(END,"Accuracy of KNN: "+str(knn_result)+"\n")
    text.insert(END,"Recall of KNN: "+str(recall_knn)+"\n")
    text.insert(END,"Precision of KNN: "+str(precision_knn)+"\n")

    ann_clf = MLPClassifier()

    #Parameters
    parameters = {'solver': ['lbfgs'],
                  'alpha':[1e-4],
                  'hidden_layer_sizes':(9,14,14,2),   # 9 input, 14-14 neuron in 2 layers,1 output layer
                  'random_state': [1]}
    # Type of scoring to compare parameter combos
    acc_scorer = make_scorer(accuracy_score)

    # Run grid search
    grid_obj = GridSearchCV(ann_clf, parameters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, y_train)

    # Pick the best combination of parameters
    ann_clf = grid_obj.best_estimator_

    # Fit the best algorithm to the data
    ann_clf.fit(X_train, y_train)
    y_pred_ann = ann_clf.predict(X_test)

    ann_result = accuracy_score(y_test,y_pred_ann)

    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 9))
    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

    # Part 3 - Making the predictions and evaluating the model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    recall_ann = recall_score(y_test,y_pred_ann)
    precision_ann = precision_score(y_test,y_pred_ann)
    text.insert(END,"Accuracy of ANN: "+str(ann_result)+"\n")
    text.insert(END,"Recall of ANN: "+str(recall_ann)+"\n")
    text.insert(END,"Precision of ANN: "+str(precision_ann)+"\n")


def graph():
    global svm_result,knn_result,ann_result
    global recall_svm,recall_knn,recall_ann
    global precision_svm,precision_knn,precision_ann

    results ={'Accuracy': [svm_result*100,knn_result*100,ann_result*100],
              'Recall': [recall_svm*100,recall_knn*100,recall_ann*100],
              'Precision': [precision_svm*100,precision_knn*100,precision_ann*100]}
    index = ['SVM','KNN','ANN']
    results =pd.DataFrame(results,index=index)
    fig =results.plot(kind='bar',title='Comaprison of models',figsize =(9,9)).get_figure()
    fig.savefig('Final Result.png')

    fig =results.plot(kind='bar',title='Comaprison of models',figsize =(6,6),ylim=[50,100]).get_figure()
    fig.savefig('image.png')

    results.plot(subplots=True,kind ='bar',figsize=(4,10))
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Heart Disease Identification Method Using Machine Learning Classification in E-Healthcare')
title.config(bg='dark salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700,y=150)

ip = Button(main, text="Data Import", command=importdata)
ip.place(x=700,y=200)
ip.config(font=font1)

pp = Button(main, text="Data Preprocessing", command=preprocess)
pp.place(x=700,y=250)
pp.config(font=font1)

tt = Button(main, text="Train and Test Model", command=ttmodel)
tt.place(x=700,y=300)
tt.config(font=font1)

ml = Button(main, text="Run Algorithms", command=models)
ml.place(x=700,y=350)
ml.config(font=font1)

gph = Button(main, text="Accuracy Graph", command=graph)
gph.place(x=700,y=400)
gph.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='ivory2')
main.mainloop()

