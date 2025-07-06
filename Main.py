from tkinter import *
import tkinter
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import warnings
warnings.filterwarnings('ignore')
import pandas as pd     
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample

import os
import pickle

from keras.callbacks import ModelCheckpoint

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Activation, Convolution2D, Embedding, GRU, SimpleRNN
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer

main = tkinter.Tk()
main.title("A deep learning approach for detecting malicious activities for mobile edge security") 
main.geometry("1000x650")


accuracy = []
precision = []
recall = []
fscore = []

# Lists to store metrics
accuracy = []
precision = []
recall = []
fscore = []

# Dataframes to store results
metrics_df = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
class_report_df = pd.DataFrame()
class_performance_dfs = {}  # Dictionary to store dataframes for each class

if not os.path.exists('results'):
    os.makedirs('results')

if not os.path.exists('model'):
    os.makedirs('model')


def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "datasets")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    dataset = pd.read_csv(filename)
    dataset.replace('?', pd.NA, inplace=True)
    dataset.dropna(inplace=True)
    
    text.insert(END,str(dataset.head())+"\n\n")

def preprocessDataset():
    global X, y
    global le
    #global dataset
    global x_train, x_test, y_train, y_test
    le = LabelEncoder()
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset.head())+"\n\n")
    
    dataset.head()
        # Encode categorical variables (Protocol and Anomaly) using LabelEncoder
    dataset['class'] = le.fit_transform(dataset['class'])
    # Split the dataset into features (X) and target variable (y)
    X = dataset.drop(['class'], axis=1)
    y = dataset['class']
    text.insert(END,"Total records found in dataset: "+str(X.shape[0])+"\n\n")
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
    text.insert(END,"Total records found in dataset to train: "+str(x_train.shape[0])+"\n\n")
    text.insert(END,"Total records found in dataset to test: "+str(x_test.shape[0])+"\n\n")
    # Create a count plot
    sns.set(style="darkgrid")  # Set the style of the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    # Replace 'dataset' with your actual DataFrame and 'Drug' with the column name
    ax = sns.countplot(x='class', data=dataset, palette="Set3")
    plt.title("Count Plot")  # Add a title to the plot
    plt.xlabel("Class Categories")  # Add label to x-axis
    plt.ylabel("Count")  # Add label to y-axis
    # Annotate each bar with its count value
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

    plt.show()  # Display the plot
  

def Calculate_Metrics(algorithm, predict, y_test):
    global metrics_df, class_report_df, class_performance_dfs
    
    #categories = np.unique(y_test).tolist()
    categories = ["Normal", "Malicious"]
    
    # Calculate overall metrics
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100

    # Append to global lists
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    # Create metrics dataframe entry
    metrics_entry = pd.DataFrame({
        'Algorithm': [algorithm],
        'Accuracy': [a],
        'Precision': [p],
        'Recall': [r],
        'F1-Score': [f]
    })
    metrics_df = pd.concat([metrics_df, metrics_entry], ignore_index=True)
    
    # Text output
    text.insert(END, f"{algorithm} Accuracy  : {a:.2f}\n")
    text.insert(END, f"{algorithm} Precision : {p:.2f}\n")
    text.insert(END, f"{algorithm} Recall    : {r:.2f}\n")
    text.insert(END, f"{algorithm} FScore    : {f:.2f}\n")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, predict)
    
    # Classification report
    CR = classification_report(y_test, predict, target_names=[str(c) for c in categories], output_dict=True)
    text.insert(END, f"{algorithm} Classification Report\n")
    text.insert(END, f"{algorithm}\n{str(classification_report(y_test, predict, target_names=[str(c) for c in categories]))}\n\n")
    
    # Classification report dataframe
    cr_df = pd.DataFrame(CR).transpose()
    cr_df['Algorithm'] = algorithm
    class_report_df = pd.concat([class_report_df, cr_df], ignore_index=False)
    
    # Class-specific performance dataframes
    for category in categories:
        class_entry = pd.DataFrame({
            'Algorithm': [algorithm],
            'Precision': [CR[str(category)]['precision'] * 100],
            'Recall': [CR[str(category)]['recall'] * 100],
            'F1-Score': [CR[str(category)]['f1-score'] * 100],
            'Support': [CR[str(category)]['support']]
        })
        
        # Initialize dataframe for this class if it doesn't exist
        if str(category) not in class_performance_dfs:
            class_performance_dfs[str(category)] = pd.DataFrame(columns=['Algorithm', 'Precision', 'Recall', 'F1-Score', 'Support'])
        
        # Append to class-specific dataframe
        class_performance_dfs[str(category)] = pd.concat([class_performance_dfs[str(category)], class_entry], ignore_index=True)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(conf_matrix, xticklabels=categories, yticklabels=categories, annot=True, cmap="viridis", fmt="g")
    ax.set_ylim([0, len(categories)])
    plt.title(f"{algorithm} Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(f"results/{algorithm.replace(' ', '_')}_confusion_matrix.png")
    plt.show()   


def existing_classifier():
    
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)

    mlmodel = LogisticRegression(C=0.01, penalty='l1',solver='liblinear')
    mlmodel.fit(x_train, y_train)
    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Existing LRC", y_pred, y_test)


def existing_classifier1():
    
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)

    mlmodel = KNeighborsClassifier(n_neighbors=500)
    mlmodel.fit(x_train, y_train)
    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Existing KNN", y_pred, y_test)
    

def existing_classifier2():
    
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)

    mlmodel = DecisionTreeClassifier(criterion = "entropy",max_leaf_nodes=2,max_features="auto")
    mlmodel.fit(x_train, y_train)

    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Existing DTC", y_pred, y_test)

    
        

def existing_classifier4():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)
    #now train LSTM algorithm
 
    mlmodel = AdaBoostClassifier()
    mlmodel.fit(x_train, y_train)

    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Existing AdaBoost", y_pred, y_test)


    
def existing_classifier5():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)
    #now train LSTM algorithm
 
    mlmodel = LinearDiscriminantAnalysis()
    mlmodel.fit(x_train, y_train)

    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Existing LDA", y_pred, y_test)

  


    
def trainDNN():
    global x_train, y_train, x_test, y_test, dnn
    text.delete('1.0', END)

    # One-hot encode categorical labels
    y_train_encoded = to_categorical(y_train, num_classes=2)
    y_test_encoded = to_categorical(y_test, num_classes=2)

    dnn = Sequential()
    dnn.add(Dense(units=128, input_shape=(x_train.shape[1],), activation='selu'))
    dnn.add(Dropout(0.5))  # Add dropout for regularization
    dnn.add(Dense(units=64, activation='relu'))
    dnn.add(Dropout(0.5))
    dnn.add(Dense(units=32, activation='relu'))
    dnn.add(Dropout(0.5))
    dnn.add(Dense(units=32, activation='relu'))
    dnn.add(Dense(units=2, activation='sigmoid'))  # Adjust the number of units
    dnn.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Model training
    if not os.path.exists("model/dnn_weights.h5"):
        model_checkpoint = ModelCheckpoint(filepath='model/dnn_weights.h5', verbose=1, save_best_only=True)
        hist = dnn.fit(x_train, y_train_encoded, batch_size=8, epochs=16, validation_data=(x_test, y_test_encoded), callbacks=[model_checkpoint], verbose=1)

        predictions = dnn.predict(x_test)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_test_encoded, axis=1)

    else:
        dnn.load_weights("model/dnn_weights.h5") 
    predictions = dnn.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test_encoded, axis=1)

    Calculate_Metrics("Proposed DNN", true_labels, predicted_labels)

  



def Predict():

    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, f'{filename} Loaded\n')
    test = pd.read_csv(filename)
    unique_labels = ["Normal", "Malicious"]
    prediction =dnn.predict(test)
    predict = np.argmax(prediction, axis=1)
    
    text.insert(END, f'Predicted Outcomes for each row:\n')
    for index, row in test.iterrows():
        predicted_index = predict[index]
      
        
        predicted_outcome = unique_labels[predicted_index]
        
        text.insert(END, f'Row {index + 1}: {row.to_dict()} - Predicted Outcome: {predicted_outcome}\n\n\n\n\n')

         
def graph():
    global metrics_df, class_report_df, class_performance_dfs
    
    # Plot overall metrics
    melted_df = pd.melt(metrics_df, id_vars=['Algorithm'], 
                        value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        var_name='Parameters', value_name='Value')
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Parameters', y='Value', hue='Algorithm', data=melted_df)
    
    plt.title('Classifier Performance Comparison', fontsize=14, pad=10)
    plt.ylabel('Score (%)', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)
    
    plt.tight_layout()
    plt.savefig('results/classifier_performance.png')
    plt.show()
    
    # Plot class-specific performance
    for class_name, class_df in class_performance_dfs.items():
        melted_class_df = pd.melt(class_df, id_vars=['Algorithm'], 
                                 value_vars=['Precision', 'Recall', 'F1-Score'],
                                 var_name='Parameters', value_name='Value')
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Parameters', y='Value', hue='Algorithm', data=melted_class_df)
        
        plt.title(f'Class {class_name} Performance Comparison', fontsize=14, pad=10)
        plt.ylabel('Score (%)', fontsize=12)
        plt.xlabel('Metrics', fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3)
        
        plt.tight_layout()
        plt.savefig(f'results/class_{class_name}_performance.png')
        plt.show()
    
    # Save dataframes to Excel
    metrics_df.to_excel('results/overall_metrics.xlsx', index=False)
    class_report_df.to_excel('results/classification_report.xlsx', index=True)
    for class_name, class_df in class_performance_dfs.items():
        class_df.to_excel(f'results/class_{class_name}_performance.xlsx', index=False)
    
        
        
def close():
    main.destroy()

font = ('times', 16, 'bold')

title = Label(
    main,
    text='A Deep Learning Approach for Detecting Malicious Activities for Mobile Edge Security',
    justify=LEFT,
    bg='pink',         # Set pink background
    fg='black',        # Set text color
    font=font,
    height=3,
    width=120
)
title.place(x=100, y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Dataset", command=uploadDataset)
uploadButton.place(x=200,y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

MLButton = Button(main, text="Existing KNN", command=existing_classifier1)
MLButton.place(x=500,y=100)
MLButton.config(font=font1)

MLButton = Button(main, text="Existing LRC", command=existing_classifier)
MLButton.place(x=700,y=100)
MLButton.config(font=font1)

MLButton = Button(main, text="Existing DTC", command=existing_classifier2)
MLButton.place(x=900,y=100)
MLButton.config(font=font1)

MLButton = Button(main, text="Existing AdaBoost", command=existing_classifier4)
MLButton.place(x=1100,y=100)
MLButton.config(font=font1)

MLButton = Button(main, text="Existing LDA", command=existing_classifier5)
MLButton.place(x=1300,y=100)
MLButton.config(font=font1)

dnnButton = Button(main, text="Proposed DNN", command=trainDNN)
dnnButton.place(x=500,y=150)
dnnButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=800,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Prediction", command=Predict)
predictButton.place(x=500,y=200)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=800,y=200)
exitButton.config(font=font1)

                            

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=170)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1) 

main.config(bg='lavender')
main.mainloop()
