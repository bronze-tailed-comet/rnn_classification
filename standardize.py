import numpy as np
from keras.callbacks import TensorBoard
from keras.preprocessing import sequence
import pickle
import networks

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

train_pad_average = np.mean( x_train_pad , axis = 0 )
train_pad_variance = np.var( x_train_pad , axis = 0 )

x_train_pad = ( x_train_pad - train_pad_average ) / np.sqrt( train_pad_variance )

x_train_pad = np.nan_to_num(x_train_pad)

x_test_pad = ( x_test_pad - train_pad_average ) / np.sqrt( train_pad_variance )

x_test_pad = np.nan_to_num(x_test_pad)

finalrnn = networks.rnn_padding_gn( nb_features , nb_labels , pad )

finalrnn.fit(x=[x_train_pad, y_train_pad, x_train_len_pad , y_train_len_pad ], y=np.zeros( len(y_train_pad) ),
             batch_size=batch_size, epochs=nb_epochs)

tb = TensorBoard( log_dir='./logs' , histogram_freq=0 , batch_size=32 )

finalrnn.fit(x=[x_train_pad, y_train_pad, x_train_len_pad , y_train_len_pad ], y=np.zeros( len(y_train_pad) ),
             batch_size=batch_size, epochs=nb_epochs , callbacks = [tb])
