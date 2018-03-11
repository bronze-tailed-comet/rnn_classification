# If you use Spyder and get this message at launch :
    # " A Message class can only inherit from Message"
    # ==> you should restart the kernel and try again

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import pickle
import networks

( x_train , y_train ) , ( x_test , y_test ) = mnist.load_data() 

print("\nApp data")
print("Number of points : %d" % y_train.shape[0])
print("Rows : %d, columns : %d" % (x_train.shape[1], x_train.shape[2]))

print("\nTest data")
print("Number of points : %d" % y_test.shape[0])
print("Lines : %d, columns : %d" % (x_test.shape[1], x_test.shape[2])) 

T = x_train.shape[1]
D = x_train.shape[2]

print("\nT =",T)
print("D =",D)

x_train = x_train / 255
x_test = x_test / 255
y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10)

nb_features = len(x_train[0][0])
nb_labels = 10

nctc = networks.rnn_sequence( nb_features , nb_labels )

( x_train , y_train ) , ( x_test , y_test ) = pickle.load( open ( "data/seqDigits.pkl" , "rb" ) )

x_train_len = np.asarray([len(x_train[i]) for i in range(len(x_train))])
x_test_len = np.asarray([len(x_test[i]) for i in range(len(x_test))])
y_train_len = np.asarray([len(y_train[i]) for i in range(len(y_train))])
y_test_len = np.asarray([len(y_test[i]) for i in range(len(y_test))])

batch_size = 32
nb_epochs = 2
Ltrain = len(x_train)
Ltest = len(x_test)
nb_features = len(x_train[0][0])

histctc = nctc.fit(x=[x_train , y_train , x_train_len, y_train_len], y=np.zeros( len(y_train) ), 
                   batch_size=batch_size, epochs=nb_epochs)

y_pred = nctc.predict( [x_test , x_test_len] )

for i in range(15): 
    print("y_pred =", [j for j in y_pred[i] if j!=-1], " -- y_test : ", y_test[i]) 

