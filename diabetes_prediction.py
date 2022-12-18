# Importing libraries

# numpy for numerical operations
import numpy as np
# pandas for data manipulation
import pandas as pd
# matplotlib for data visualization
import matplotlib.pyplot as plt
# seaborn for statistical data visualization
import tkinter as tk

# standard scaler for data standardization
from sklearn.preprocessing import StandardScaler
# train test split for splitting data into training and testing set
from sklearn.model_selection import train_test_split
# SVM classifier
from sklearn import svm
# Logistic Regression classifier
from sklearn import linear_model
# Naive Bayes classifier GaussianNB
from sklearn.naive_bayes import GaussianNB
# decision tree classifier
from sklearn import tree
# accuracy score for model evaluation
from sklearn.metrics import accuracy_score
# confusion matrix for model evaluation
from sklearn.metrics import confusion_matrix
# plot confusion matrix for model evaluation
from sklearn.metrics import plot_confusion_matrix
# confusion matrix display for model evaluation
from sklearn.metrics import ConfusionMatrixDisplay


# PIL for image processing and display
from PIL import Image,ImageTk


#loading ML dataset to pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')


# grouping data by outcome and calculating mean
diabetes_dataset.groupby('Outcome').mean()


# Separating Data and Labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

#Data Standardization


scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['Outcome']

#Train Test Split

# Stratify parameter makes a split so that the proportion of values
#  in the sample produced will be the same as the proportion of values provided to parameter stratify.
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


#Training the Model
classifier = svm.SVC(kernel='linear')
classifier2 = linear_model.LogisticRegression()
classifier3 = GaussianNB()
classifier4 = tree.DecisionTreeClassifier()

#Training Classifier SVM

classifier.fit(X_train, Y_train)

#Model Evaluation
#Accuracy Score

#On Training data

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('SVM Accuracy score on the training data : ', training_data_accuracy)

#On Test data

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('SVM Accuracy score on the test data : ', test_data_accuracy)
plot_confusion_matrix(classifier, X_test, Y_test) 
plt.show()

#Training Classifier Logistic Regression

classifier2.fit(X_train, Y_train)

#Model Evaluation
#Accuracy Score

#On Training data

X_train_prediction = classifier2.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('LogisticRegression Accuracy score on the training data : ', training_data_accuracy)

#On Test data

X_test_prediction = classifier2.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('LogisticRegression Accuracy score on the test data : ', test_data_accuracy)
plot_confusion_matrix(classifier2, X_test, Y_test) 
plt.show()

#Training Classifier GNB

classifier3.fit(X_train, Y_train)

#Model Evaluation
#Accuracy Score

#On Training data

X_train_prediction = classifier3.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('GNB Accuracy score on the training data : ', training_data_accuracy)

#On Test data

X_test_prediction = classifier3.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('GNB Accuracy score on the test data : ', test_data_accuracy)
plot_confusion_matrix(classifier3, X_test, Y_test) 
plt.show()

#Training Classifier Decision Tree

classifier4.fit(X_train, Y_train)

#Model Evaluation
#Accuracy Score

#On Training data

X_train_prediction = classifier4.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Decision Tree Accuracy score on the training data : ', training_data_accuracy)

#On Test data

X_test_prediction = classifier4.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Decision Tree Accuracy score on the test data : ', test_data_accuracy)
plot_confusion_matrix(classifier4, X_test, Y_test) 
plt.show()

#Prediction System with GUI

Pregnancies = 6
Glucose = 148
BloodPressure = 72
SkinThickness = 35
Insulin = 0
BMI = 33.6
DiabetesPedigreeFunction = 0.627
Age = 50	



#Prediction Function

def predFunction(algorithm):
    print(algorithm)
    # 1.0 reading from the first letter of the first line
    # end-1c reading from the first letter of the first line to the last letter of the last line
    Pregnancies = int(inputtxt.get("1.0","end-1c"))
    Glucose = float(inputtxt2.get("1.0","end-1c"))
    BloodPressure = float(inputtxt3.get("1.0","end-1c"))
    SkinThickness = float(inputtxt4.get("1.0","end-1c"))
    Insulin = float(inputtxt5.get("1.0","end-1c"))
    BMI = float(inputtxt6.get("1.0","end-1c"))
    DiabetesPedigreeFunction = float(inputtxt7.get("1.0","end-1c"))
    Age = int(inputtxt8.get("1.0","end-1c"))

    input_data = (Pregnancies,Glucose,BloodPressure,SkinThickness
    ,Insulin,BMI,DiabetesPedigreeFunction,Age)

# changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
    std_data = scaler.transform(input_data_reshaped)


    if(algorithm == "Support Vector Machine" ):
        prediction = classifier.predict(std_data)
    if(algorithm == "Logistic Regression" ):
        prediction = classifier2.predict(std_data)
    if(algorithm == "Naive Bayes" ):
        prediction = classifier3.predict(std_data)
    if(algorithm == "Decision Tree" ):
        prediction = classifier4.predict(std_data)    
        
    print(prediction)
    
    if (prediction[0] == 0):
      t10['text'] = "Not Diabetic"
    else:
      t10['text'] = "Diabetic"
    


#User Interface (GUI)

root = tk.Tk()
#Title

root.title("Diabetes Predictor")
root.configure(background = "#ffa51d")
canvas = tk.Canvas(root, width=500, height=300 )
canvas.grid()

#Logo

logo = Image.open('newDiabetic.jpg')
logo = logo.resize((500,350), Image.ANTIALIAS)
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(image=logo)
logo_label.image = logo
logo_label.grid(sticky="W",columnspan=4,column=0,row=0)

#Algorithm Selection dropdown

instruction = tk.Label(root, text="Select an Algorithm and Fill the Following Fields" ,background = "#ffa51d" , font = "Helvetica 12 bold ")
instruction.grid(sticky="W",columnspan=4,column=0, row=1,padx=45,pady=5)

menu= tk.StringVar()
menu.set("Select Algorithm")

#Create a dropdown Menu

drop= tk.OptionMenu(root ,menu,"Support Vector Machine", "Logistic Regression","Naive Bayes","Decision Tree" )
drop.config(width=30 , background = "#ffa51d" , font = "Helvetica 8 bold " )
drop.grid(sticky="W",columnspan=3,column=0, row=2,padx=35,pady=8 )


#Input Fields

t1 = tk.Label(root, text="Pregnancies:",background = "#ffa51d" , font = "Helvetica 8 bold ")
t1.grid(sticky="W",column=0,row=3,pady=2,padx=8)

inputtxt = tk.Text(root,
                   height = 1.2,
                   width = 40)
  
inputtxt.grid(columnspan=2,column=0, row=3 ,pady=2)

t2 = tk.Label(root, text="Glucose:" ,background = "#ffa51d" , font = "Helvetica 8 bold ")
t2.grid(sticky="W",column=0,row=4,pady=2,padx=8)

inputtxt2 = tk.Text(root,
                   height = 1.2,
                   width = 40)
  
inputtxt2.grid(columnspan=2,column=0, row=4,pady=2)


t3 = tk.Label(root, text="BP:" ,background = "#ffa51d" , font = "Helvetica 8 bold ")
t3.grid(sticky="W",column=0,row=5,pady=2,padx=8)

inputtxt3 = tk.Text(root,
                   height = 1.2,
                   width = 40)
  
inputtxt3.grid(columnspan=2,column=0, row=5,pady=2)

t4 = tk.Label(root, text="ST:" ,background = "#ffa51d" , font = "Helvetica 8 bold ")
t4.grid(sticky="W",column=0,row=6,pady=2 ,padx=8)

inputtxt4 = tk.Text(root,
                   height = 1.2,
                   width = 40)
  
inputtxt4.grid(columnspan=2,column=0, row=6,pady=2)

t5 = tk.Label(root, text="Insulin:" ,background = "#ffa51d" , font = "Helvetica 8 bold ")
t5.grid(sticky="W",column=0,row=7,pady=2,padx=8)

inputtxt5 = tk.Text(root,
                   height = 1.2,
                   width = 40)
  
inputtxt5.grid(columnspan=2,column=0, row=7,pady=2)

t6 = tk.Label(root, text="BMI:" ,background = "#ffa51d" , font = "Helvetica 8 bold ")
t6.grid(sticky="W",column=0,row=8,pady=2,padx=8)

inputtxt6 = tk.Text(root,
                   height = 1.2,
                   width = 40)
  
inputtxt6.grid(columnspan=2,column=0, row=8,pady=2)

t7 = tk.Label(root, text="DPF:" ,background = "#ffa51d" , font = "Helvetica 8 bold ")
t7.grid(sticky="W",column=0,row=9,pady=2,padx=8)

inputtxt7 = tk.Text(root,
                   height = 1.2,
                   width = 40)
  
inputtxt7.grid(columnspan=2,column=0, row=9,pady=2)


t8 = tk.Label(root, text="Age:" ,background = "#ffa51d" , font = "Helvetica 8 bold ")
t8.grid(sticky="W",column=0,row=10,pady=2,padx=8)

inputtxt8 = tk.Text(root,
                   height = 1.2,
                   width = 40)
  
inputtxt8.grid(columnspan=2,column=0, row=10,pady=2)

t9 = tk.Label(root, text="Prediction:" ,background = "#ffa51d" , font = "Helvetica 8 bold ")
t9.grid(sticky="W",column=0,row=11,pady=2, padx=8)

t10 = tk.Label(root, text="--" ,background = "#ffa51d" , font = "Helvetica 8 bold ")
t10.grid(columnspan=2,column=0,row=11,pady=2)

#Prediction Button

button_txt = tk.StringVar()
button_txt.set("Predict")
button = tk.Button(root,textvariable=button_txt,command=lambda:predFunction(menu.get()),height=2,width=30 ,background = "#ffa51d" , font = "Helvetica 8 bold", activebackground = "#e65239")
button.grid(column=0,row=12 ,pady=2)

#Copyrights

# t11 = tk.Label(root, text="Developed by:\nSaad Sajjad(SP19-BCS-044)\nAbdullah Khalid Pasha(SP19-BCS-141)\nHamid Jamil(SP19-BCS-098)")
# t11.grid(columnspan=1,column=0,row=13)

root.mainloop()





