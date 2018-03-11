from keras.engine import Model
from keras.layers import ( Bidirectional, Masking, GaussianNoise, 
                        Input, SimpleRNN, LSTM, Dense, Dropout, TimeDistributed )
from keras.optimizers import RMSprop, Adam
from CTCModel import CTCModel

def classif_rnn_network(T , D):

    input_layer = Input( shape=( T , D ) , batch_shape=None )

    simplernn_layer = SimpleRNN( units = 20 )(input_layer)

    dense_layer = Dense( units = 10 , activation = "softmax" )(simplernn_layer)
    
    network = Model( inputs = input_layer , outputs = dense_layer )

    network.compile( optimizer = "sgd" , loss = "categorical_crossentropy" , metrics = ["accuracy"] )
    
    return network 


def improved_classif_rnn_network(T , D):

    inp = Input(shape=( T , D ))

    cc1 = LSTM( units = 100 , activation = "tanh" , return_sequences=True)(inp)

    cc2 = LSTM(units = 100, activation='tanh')(cc1)
    
    d = Dense( units = 10 , activation = "softmax" )(cc2)
    
    network = Model( inputs = inp , outputs = d )

    network.compile( optimizer = "sgd" , loss = "categorical_crossentropy" , metrics = ["accuracy"] )
    
    return network


def rnn_sequence( nb_features, nb_labels ):
    
    input_layer = Input(shape=(None, nb_features)) 

    HL_1 = Bidirectional( LSTM( 16 , return_sequences = True ) )(input_layer)
    HL_1D = Dropout(0.1)(HL_1)
    
    HL_2 = Bidirectional( LSTM( 16 , return_sequences = True ) )(HL_1D)
    HL_2D = Dropout(0.1)(HL_2)
    
    HL_3 = Bidirectional( LSTM( 16 , return_sequences = True ) )(HL_2D)
    HL_3D = Dropout(0.1)(HL_3)

    output = TimeDistributed(Dense(nb_labels + 1, activation = "softmax"))(HL_3D)
    
    network = CTCModel([input_layer], [output])
    network.compile(RMSprop(lr=0.01))

    return network


def rnn_padding( nb_features , nb_labels , pad ): # dealing with variable length sequences
    
    input_layer = Input(shape=(None, nb_features)) 
    
    mask = Masking(mask_value=pad)(input_layer)

    HL_1 = Bidirectional( LSTM( 16 , return_sequences = True ) )(mask)
    HL_1D = Dropout(0.1)(HL_1)
    
    HL_2 = Bidirectional( LSTM( 16 , return_sequences = True ) )(HL_1D)
    HL_2D = Dropout(0.1)(HL_2)
    
    HL_3 = Bidirectional( LSTM( 16 , return_sequences = True ) )(HL_2D)
    HL_3D = Dropout(0.1)(HL_3)

    output = TimeDistributed(Dense(nb_labels + 1, activation = "softmax"))(HL_3D)
    
    network = CTCModel([input_layer], [output])
    network.compile(RMSprop(lr=0.01))

    return network


def rnn_padding_rmsprop( nb_features , nb_labels , pad ):  # Use RMSPROP
    
    input_layer = Input(shape=(None, nb_features)) 
    
    mask = Masking(mask_value=pad)(input_layer)

    HL_1 = Bidirectional( LSTM( 16 , return_sequences = True ) )(mask)
    HL_1D = Dropout(0.1)(HL_1)
    
    HL_2 = Bidirectional( LSTM( 16 , return_sequences = True ) )(HL_1D)
    HL_2D = Dropout(0.1)(HL_2)
    
    HL_3 = Bidirectional( LSTM( 16 , return_sequences = True ) )(HL_2D)
    HL_3D = Dropout(0.1)(HL_3)

    output = TimeDistributed(Dense(nb_labels + 1, activation = "softmax"))(HL_3D)
    
    network = CTCModel([input_layer], [output])
    network.compile(RMSprop(lr=0.01))

    return network


def rnn_padding_adam( nb_features , nb_labels , pad ):  # Use adam
    
    input_layer = Input(shape=(None, nb_features)) 
    
    mask = Masking(mask_value=pad)(input_layer)

    HL_1 = Bidirectional( LSTM( 16 , return_sequences = True ) )(mask)
    HL_1D = Dropout(0.1)(HL_1)
    
    HL_2 = Bidirectional( LSTM( 16 , return_sequences = True ) )(HL_1D)
    HL_2D = Dropout(0.1)(HL_2)
    
    HL_3 = Bidirectional( LSTM( 16 , return_sequences = True ) )(HL_2D)
    HL_3D = Dropout(0.1)(HL_3)

    output = TimeDistributed(Dense(nb_labels + 1, activation = "softmax"))(HL_3D)
    
    network = CTCModel([input_layer], [output])
    network.compile(Adam(lr=0.01))

    return network


def rnn_padding_gn( nb_features , nb_labels , pad ):  # Use RMSPROP and add Gaussian noise
    
    input_layer = Input(shape=(None, nb_features)) 
    
    mask = Masking(mask_value = pad)(input_layer)

    gn = GaussianNoise(0.01)(mask)

    HL_1 = Bidirectional( LSTM( 16 , activation = "tanh" , return_sequences = True ) )(gn)
    HL_1D = Dropout(0.1)(HL_1)
    
    HL_2 = Bidirectional( LSTM( 16 , activation = "tanh" , return_sequences = True ) )(HL_1D)
    HL_2D = Dropout(0.1)(HL_2)
    
    HL_3 = Bidirectional( LSTM( 16 , activation = "tanh" , return_sequences = True ) )(HL_2D)
    HL_3D = Dropout(0.1)(HL_3)

    output = TimeDistributed(Dense(nb_labels + 1, activation = "softmax"))(HL_3D)
    
    network = CTCModel([input_layer], [output])
    network.compile( RMSprop ( lr = 0.01 ) )

    return network


