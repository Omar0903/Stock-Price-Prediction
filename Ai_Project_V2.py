# Importing all Needed Libraries
from tkinter import *
from tkinter import ttk
from tkinter import Tk
import pandas as pd
import numpy as py
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
def predict():
    # cleaning data
    df = pd.read_csv('stock.csv')
    data=df.drop(columns=["Unnamed: 6","Unnamed: 7","Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13","Unnamed: 14"],axis=1)
    data['Date'] = pd.to_datetime(data['Date'])
    data.duplicated()
    data.replace("down",0,inplace=True)
    data.replace("flat",1,inplace=True)
    data.replace("up",2,inplace=True)
    #Replace NULL with mean values
    data['Date'].fillna(data['Date'].mean(),inplace=True)
    data['AMZN'].fillna(data['AMZN'].mean(),inplace=True)
    data["DPZ"].fillna(data["DPZ"].mean(),inplace=True)
    data["BTC"].fillna(data["BTC"].mean(),inplace=True)
    data["NFLX"].fillna(data["NFLX"].mean(),inplace=True)
    #Replace NULL with median values
    # data['Date'].fillna(data['Date'].median(),inplace=True)
    # data['AMZN'].fillna(data['AMZN'].median(),inplace=True)
    # data["DPZ"].fillna(data["DPZ"].median(),inplace=True)
    # data["BTC"].fillna(data["BTC"].median(),inplace=True)
    # data["NFLX"].fillna(data["NFLX"].median(),inplace=True)
    # SVM model
    if cmbo1.get()== "SVM" :
        x = data.drop(columns=['Date','Price Movement '])
        y = data["Price Movement "]
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
        model=svm.SVC()
        model.fit(x_train,y_train)
        y_predict=model.predict(x_test)
        TestAccuracy = metrics.accuracy_score(y_test, y_predict)
        model.fit(x_train,y_train)
        y_predict=model.predict(x_train)
        TrainAccuracy = metrics.accuracy_score(y_train, y_predict)
        user_input=py.array([int(En1.get()),float(En2.get()),float(En3.get()),float(En4.get())])
        user_input=py.expand_dims(user_input,axis=0)
        user_prediction=model.predict(user_input)
        NewPredict = user_prediction
        if NewPredict == 0 :
            NewPredict = "Down"
        elif NewPredict == 1 :
            NewPredict = "flat"
        else :
            NewPredict = "UP"
    elif cmbo1.get()=="DT":
        X = data.drop(columns=['Date','Price Movement '])
        Y= data['Price Movement ']
        # #Splitting data into train and test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
        # #Decisin Tree model
        acc1=[]
        acc2=[]
        for i in range (2,16):
            dtc1=DecisionTreeClassifier(max_depth=i,criterion="entropy",random_state=42)
            dtc1.fit(X_train,Y_train)
            dtc2=DecisionTreeClassifier(max_depth=i,criterion="gini",random_state=42)
            dtc2.fit(X_train,Y_train)
            acc1.append(dtc1.score(X_test, Y_test))
            acc2.append(dtc2.score(X_test, Y_test))
        clf = DecisionTreeClassifier(criterion='entropy', max_depth = 4 , random_state = 42)
        clf = clf.fit(X_train , Y_train)
        Y_pred_test = clf.predict(X_test)
        Y_pred_train = clf.predict(X_train)
        accuracy_train = accuracy_score(Y_train, Y_pred_train) * 100  # Calculate accuracy as a percentage
        accuracy_test = accuracy_score(Y_pred_test, Y_test) * 100
        TestAccuracy = accuracy_test
        TrainAccuracy = accuracy_train
        user_input = py.array([float(En1.get()),float(En2.get()),float(En3.get()),float(En4.get())])
        user_input = py.expand_dims(user_input,axis=0)
        user_prediction = clf.predict(user_input)
        NewPredict =  user_prediction
        if NewPredict == 0 :
                NewPredict = "Down"
        elif NewPredict == 1 :
                NewPredict = "flat"
        else :
                NewPredict = "UP"
        scores_dt = cross_val_score(clf, X_train, Y_train, cv=10,scoring="accuracy")
    else :
        # Logistic Regression
        x = data.drop(columns=['Date','Price Movement '])
        y=data["Price Movement "]
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
        logisticRegModel = LogisticRegression(solver="newton-cholesky", C=1000, random_state=0)
        logisticRegModel.fit(x_train, y_train)
        logisticRegModelTrainPredication = logisticRegModel.predict(x_train) 
        logisticRegModelTestPredication = logisticRegModel.predict(x_test)
        logisticRegModelTrainAccuracy = metrics.accuracy_score(y_train, logisticRegModelTrainPredication)
        logisticRegModelTestAccuracy  = metrics.accuracy_score(y_test, logisticRegModelTestPredication)
        user_input = py.array([float(En1.get()),float(En2.get()),float(En3.get()),float(En4.get())])
        user_input = py.expand_dims(user_input,axis=0)
        myprediction = logisticRegModel.predict(user_input)      
        TrainAccuracy= logisticRegModelTrainAccuracy
        TestAccuracy = logisticRegModelTestAccuracy
        NewPredict = myprediction 
        if NewPredict == 0 :
              NewPredict = "Down"
        elif NewPredict == 1 :
             NewPredict = "flat"
        else :
             NewPredict = "UP"
    PredictScreen = Tk ()
    PredictScreen.geometry('1200x720')
    PredictScreen.title('Ai Project')
    PredictScreen.iconbitmap("C:\\Users\\Lenovo\\Desktop\\AI project\\AI.ico")
    PredictScreen.config(background='silver')
    bt2 = Button(PredictScreen,text='exit',fg='black',bg='white',width=30,height=2,command=PredictScreen.quit)
    bt2.place(x=950,y=650)
    Lb7 = Label(PredictScreen,text=' predectition ',fg='black',bg='white',font=20,width=30)
    Lb7.place(x=40,y=80)
    Lb8 = Label(PredictScreen,text=NewPredict,fg='black',bg='white',font=20)
    Lb8.place(x=450,y=80)       
    Lb9 = Label(PredictScreen,text='general predection train',fg='black',bg='white',font=20,width=30)
    Lb9.place(x=40,y=120)       
    Lb10 = Label(PredictScreen,text=' general predection test ',fg='black',bg='white',font=20,width=30)
    Lb10.place(x=40,y=160)   
    Lb11= Label(PredictScreen,text=TrainAccuracy,fg='black',bg='white',font=20)
    Lb11.place(x=450,y=120)  
    Lb12 = Label(PredictScreen,text=TestAccuracy,fg='black',bg='white',font=20)
    Lb12.place(x=450,y=160)  
    Button.pack()
    PredictScreen.mainloop
MainScreen = Tk()
MainScreen.geometry('1200x720')
MainScreen.resizable(False , False)
MainScreen.title('Ai Project')
MainScreen.iconbitmap("AI.ico")
MainScreen.config(background='silver')
fr1  = Frame(width='1150',height='470',bg='white')
fr2  = Frame(fr1,width='350',height='230',bg='silver')
fr3  = Frame(fr1,width='300',height='230',bg='silver')
Lb1  = Label(fr1,text=' Welcome to stock predectition ',fg='black',bg='silver',font=25,width=30)
Lb2  = Label(fr2,text=' Amazon stock price ',fg='black',bg='white',font=15,width=25)
Lb3  = Label(fr2,text=' Netflix stock price  ',fg='black',bg='white',font=15,width=25)
Lb4  = Label(fr2,text=" Domino's Pizza stock price ",fg='black',bg='white',font=15,width=25)
Lb5  = Label(fr2,text=' BTC stock price stock price ',fg='black',bg='white',font=15,width=25)
Lb6  = Label(text=' choose model',fg='black',bg='silver',font=15,width=25)
Lb13 = Label(text =' © 2024 AI project.All Rights Reserved ',fg='black',bg='silver',font=15)
Lb14 = Label(fr2,text =' Stock Name ',fg='black',bg='white',font=15,width=25)
Lb15 = Label(fr3,text =' Stock Price ',fg='black',bg='white',font=15,width=20)
bt1  = Button(MainScreen,text='Predict',fg='black',bg='silver',width=30,height=2,command=predict)
bt1.place(x=930,y=430)
En1 = Entry(fr3,fg='black',bg='white',font= 15)
En2 = Entry(fr3,fg='black',bg='white',font= 15)
En3 = Entry(fr3,fg='black',bg='white',font= 15)
En4 = Entry(fr3,fg='black',bg='white',font= 15)
fr1.place(x=25,y=20)
fr2.place(x=30,y=80)
fr3.place(x=400,y=80)
Lb1.place(x=360,y=20)
Lb2.place(x=40,y=60)
Lb3.place(x=40,y=100)
Lb4.place(x=40,y=140)
Lb5.place(x=40,y=180)
Lb6.place(x=95,y=350)
Lb13.place(x=800,y=680)
Lb14.place(x=40,y=20)
Lb15.place(x=40,y=20)
En1.place(x=40,y=60)
En2.place(x=40,y=100)
En3.place(x=40,y=140)
En4.place(x=40,y=180)
cmbo1 = ttk.Combobox(MainScreen, value = ('SVM','DT','Logistic Regression'))
cmbo1.set('SVM')
cmbo1.place(x=450,y=350,height=30)
MainScreen.mainloop()

