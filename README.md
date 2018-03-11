# RNN_classification
Differents ways of making a RNN for classification 

Recurrent Neural Network (RNN) in Keras in a classification task and a sequence labeling task. 
We experiment RNN in a classification task, i.e. where we have in input of our network a data represented as a time sequence and in output a unique label for the entire sequence. Here, experiments will be performed on the MNIST dataset which contains images of handwritten digits.
A recurrent neural network is defined for dealing with sequential data by making use of the temporal context during the process. In addition to connections between neurons from different layers, there are connections between neurons of the same hidden layer.

For this second part, we will make use of the Connectionist
Temporal Classification (CTC) for training the RNN in case of label sequences

Please first install the packages in the requirements (requirements.txt) : Keras, Tensorflow and pickle
	=> "pip install keras"
	=> "pip install tensorflow"
	=> "pip install pickle"

Downlad the data in pkl format with this link: 
https://www.dropbox.com/sh/6j48ti0diwrki8b/AACFmOq1Q0JZ_KVyhu62ekTwa?dl=0

In the RNN_classification folder, create a folder "data" and put the two pkl files inside.

You now can use the .py files.
There are several parts :

- classic.py :
One will define a function classif_rnn_network that build the computation graph related to a RNN for a classification task. Here, all sequences have the same length. The function has two arguments: T, the length of the sequences, and D, the number of features. First, we build a standard RNN using only one recurrent layer. Then we improve the model with more depth and strongest types of layers.
Then we try to improve your network by:
	-Increasing the number of units,
	-Adding other layers 
	-Replacing the SimpleRNN layer by GRU or LSTM layers,
	-Increasing the number of training epochs.
We can get an accuracy which is over than 95%, even if it can take a long time.

- seq_labelling.py :
The Connectionist Temporal Classification (CTC) is defined to perform a sequence labeling, i.e. where the label sequence in output can be shorter than the number of time frames of the observation sequence. This is a realistic task as it is not common to have one label per time frame. This approach has some similarities with the well-known Hidden Markov Models, such as the Forward-Backward process for instance. CTC relies on a cost function computed on the entire sequence that one will call ctc_loss to the next.
First, define a new function rnn_seqlab_network to define a recurrent neural network for sequence labeling. Make a copy of the RNN defined for a classification task, one will modify it to perform the CTC. Then we experiment the network on a sequence of digits, with images from the MNIST dataset that have been concatenated to form sequences of 5 digits.


- variable_length_seq.py :
The way for dealing with variable length sequences and fixed-size tensors in Keras and Tensorflow is to pad sequences to the longest one of the batch and to not considered the padded value using a specific layer called Masking.

- standardize.py :
Standardize is a common way for preprocessing input data. This allows to made the features uniform and to get input values closed to zero (the values follow a Gaussian of mean 0 and variance 1). 
Then we use Tensorboard as a visualization tool provided with TensorFlow. It is defined in keras.callbacks. It consists in writing a log for TensorBoard, which allows to visualize especially the computation graph and evaluation metrics. 



