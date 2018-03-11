import numpy as np
from keras.preprocessing import sequence
import networks
import pickle

pad = 255
nb_labels = 10

( x_train_pad , y_train_pad ) , ( x_test_pad , y_test_pad ) = pickle.load( open ( "data/seqDigitsVar.pkl" , "rb" ) )

x_train_pad = sequence.pad_sequences(x_train_pad, value=float(pad), dtype='float32',
                                         padding="post", truncating='post')

x_test_pad = sequence.pad_sequences(x_test_pad, value=float(pad), dtype='float32',
                                        padding="post", truncating='post')

y_train_pad = sequence.pad_sequences(y_train_pad, value=float(nb_labels),
                                         dtype='float32', padding="post")

y_test_pad = sequence.pad_sequences(y_test_pad, value=float(nb_labels),
                                        dtype='float32', padding="post")

x_train_len_pad = np.asarray([len(x_train_pad[i]) for i in range(len(x_train_pad))])
x_test_len_pad = np.asarray([len(x_test_pad[i]) for i in range(len(x_test_pad))])
y_train_len_pad = np.asarray([len(y_train_pad[i]) for i in range(len(y_train_pad))])
y_test_len_pad = np.asarray([len(y_test_pad[i]) for i in range(len(y_test_pad))])

nb_features = len(x_train_pad[0][0])

npad = networks.rnn_padding( nb_features , nb_labels , pad )

batch_size = 32
nb_epochs = 5

npad.fit( x=[x_train_pad , y_train_pad , x_train_len_pad , y_train_len_pad], y=np.zeros( len(y_train_pad) ), 
         batch_size = batch_size , epochs = nb_epochs )

y_pred = npad.predict( [x_test_pad , x_test_len_pad] )


finalrnn = networks.rnn_padding_gn( nb_features , nb_labels , pad )

nb_epochs = 5
batch_size = 32

finalrnn.fit(x=[x_train_pad, y_train_pad, x_train_len_pad , y_train_len_pad ], y=np.zeros( len(y_train_pad) ),
             batch_size=batch_size, epochs=nb_epochs)

y_pred = finalrnn.predict( [x_test_pad , x_test_len_pad] )

evaluation = finalrnn.evaluate(x=[x_test_pad, y_test_pad, x_test_len_pad, y_test_len_pad],\
                            batch_size=batch_size, metrics=['accuracy','loss', 'ler', 'ser'])
print(evaluation)
