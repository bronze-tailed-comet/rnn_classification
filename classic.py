from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

import networks

( x_train , y_train ) , ( x_test , y_test ) = mnist.load_data() 

print("\nDonnees apprentissage")
print("Nb de points : %d" % y_train.shape[0])
print("Lignes : %d, colonnes : %d" % (x_train.shape[1], x_train.shape[2]))

print("\nDonnees test")
print("Nb de points : %d" % y_test.shape[0])
print("Lignes : %d, colonnes : %d" % (x_test.shape[1], x_test.shape[2])) 

T = x_train.shape[1]
D = x_train.shape[2]

print("\nT =",T)
print("D =",D)

n = networks.classif_rnn_network( T , D )
print(n.summary())

x_train = x_train / 255
x_test = x_test / 255
y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10)

n.fit( x_train , y_train )

score_test = n.evaluate( x_test , y_test ) 

print("Loss :",score_test[0])
print("Accuracy :",score_test[1])


n.fit( x_train , y_train )

score_test = n.evaluate( x_test , y_test ) 

print("Loss :",score_test[0])
print("Accuracy :",score_test[1])

n = networks.improved_classif_rnn_network( T , D )
print(n.summary())

early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=2, verbose=0)

nbepochs = 15

impn = networks.improved_classif_rnn_network( T , D )

hist = impn.fit( x_train , y_train , 
                validation_split=0.25 , 
                epochs=nbepochs, 
                callbacks = [early_stop])

score_test = impn.evaluate( x_test , y_test )

print(score_test)













