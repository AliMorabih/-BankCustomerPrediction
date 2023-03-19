# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

# Part 1 - Data Preprocessing
dataset = pd.read_csv('/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)

print(y)

# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# We replace the column 3 the one that Gender by 0 or 1 we call this encoding, we use the method the fit_transfert methode call from le object instance of the labelencoder class
X[:, 2] = le.fit_transform(X[:, 2])

# test the encoding 
print(X)

# We need to use the On Hote coding for Geography 
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# We will have to replace 0 by 1 because our Geography column is on the 1 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building ANN 
# Initializing the ANN 
# Adding the input layer and the first hidder layer
# Adding the hidden layer
# Addint the output layer


# Initializing the ANN // with calling seq class to call tenserFlow from keras library and from models mudule 
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
# call add methode from sequential class 
# add Dense class  from layer module 
# the Input layer is the data is our features exemple (Credit score , gender, Geography, Salary....)
# this is the hidden neural layer  units=6 
# the our activation function is relu rectifile 
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# Adding the second hidden layer
# this is the hidden neural layer  units=6 
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
# the units is 1 because we have only one output 1 or 0 (mean will stay or leave the bank)
# for the activation fonction of the output layer we need sigmoid for the activation f , because its allow the prediction and also give probability 1 or 0 that why we use the Sigmoid for the output layer  
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



# Part 3 - Training the ANN
# Compiling the ANN
# The Optimizer is Adam the best in stockastique gradien descent 
# stockastique gradien descent  definition is update the weight in order to reduce the loss error between prediction and real result 
# we are doing binary classification that why the loss fonction binary_crossentropy
# If we are doing non banary Classification we need to use categorial_crossentropy for the loss and  also the activation should be softmax 
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Training the ANN on the Training set
# we need to call our object ann, as we know the methode to train is the machine learning modele is the fit  
# we using the batch size  parameter is 32, because batch learning is more efficancy  when traning Artifical Neural 
# Neraul net need a good amount of epochs to learn properly
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# if is > 0.5 prediction will yes or no mean // False mean customer will not leave the bank  

print(ann.predict(sc.transform([[1, 0, 0, 300, 1, 40, 3, 000, 1, 1, 0, 1000]])) > 0.5)

print(ann.predict(sc.transform([[1, 0, 0, 700, 1, 10, 1, 6000, 1, 0, 0, 150000]])))




