

from __future__ import print_function
import numpy as np
import keras
from keras.utils import np_utils
import matplotlib.pyplot as plt
np.random.seed(42)

class Layer:
    def __init__(self):
        pass
    
    def feed_forward(self, input):
        return input

    def backward(self, input, grad_output):
        num_units = input.shape[1]
        
        d_layer_d_input = np.eye(num_units)
        
        return np.dot(grad_output, d_layer_d_input)

class ReLU(Layer):
    def __init__(self):
        pass
    
    def feed_forward(self, input):
        relu_feed_forward = np.maximum(0,input)
        return relu_feed_forward
    
    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(2/(input_units+output_units)), 
                                        size = (input_units,output_units))
        self.biases = np.zeros(output_units)
        
    def feed_forward(self,input):
        return np.dot(input,self.weights) + self.biases
    
    def backward(self,input,grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        # SGD step. 
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        
        return grad_input

def softmax_crossentropy(logits,reference_answers):
    logits_for_answers = logits[np.arange(len(logits)),reference_answers]
    
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
    
    return xentropy

def grad_softmax_crossentropy(logits,reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)),reference_answers] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- ones_for_answers + softmax) / logits.shape[0]



def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.reshape(y_train.shape[0], )
    y_test = y_test.reshape(y_test.shape[0], )
    # normalize x
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255

    # cross validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

network = []
network.append(Dense(X_train.shape[1],1024))
network.append(ReLU())
network.append(Dense(1024, 512))
network.append(ReLU())
network.append(Dense(512, 256))
network.append(ReLU())
network.append(Dense(256, 10))

def feed_forward(network, X):
    activations = []
    input = X

    for l in network:
        activations.append(l.feed_forward(input))
        input = activations[-1]
    
    assert len(activations) == len(network)
    return activations

def predict(network,X):
    logits = feed_forward(network,X)[-1]
    return logits.argmax(axis=-1)

def train(network,X,y):
    layer_activations = feed_forward(network,X)
    layer_inputs = [X]+layer_activations  
    logits = layer_activations[-1]
    loss = softmax_crossentropy(logits,y)
    loss_grad = grad_softmax_crossentropy(logits,y)
    
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        
        loss_grad = layer.backward(layer_inputs[layer_index],loss_grad) 
        
    return np.mean(loss)

from tqdm import trange
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

train_log = list()
val_log = list()

#for epoch in range(25):
#
#    for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=128,shuffle=True):
#        train(network,x_batch,y_batch)
#    
#    train_log.append(np.mean(predict(network,X_train)==y_train))
#    val_log.append(np.mean(predict(network,X_val)==y_val))
#    print("Epoch",epoch)
#    print("Train accuracy:",train_log[-1])
#    print("Val accuracy:",val_log[-1])
#    
#
#plt.plot(train_log,label='train accuracy')
#plt.plot(val_log,label='val accuracy')
#plt.legend(loc='best')
#plt.grid()
#plt.show()



"""#playing with hyper-parameters"""

#network = []
#number_of_layers = [2, 3, 4]
#number_of_units = [[512, 256, 128, 64], [256, 128, 64, 32], [128, 64, 32, 16], [1024, 512, 256, 128]]
#best_accuracy = 0
#best_num_layers = 0
#best_num_units = []
#
#for i in number_of_layers:
#  for j in number_of_units:
#    if i==2:
#      network = []
#      network.append(Dense(X_train.shape[1], j[0]))
#      network.append(ReLU())
#      network.append(Dense(j[0], j[1]))
#      network.append(ReLU())
#      network.append(Dense(j[1], 10))
#    elif i==3:
#      network = []
#      network.append(Dense(X_train.shape[1], j[0]))
#      network.append(ReLU())
#      network.append(Dense(j[0], j[1]))
#      network.append(ReLU())
#      network.append(Dense(j[1], j[2]))
#      network.append(ReLU())
#      network.append(Dense(j[2], 10))
#    elif i==4:
#      network = []
#      network.append(Dense(X_train.shape[1], j[0]))
#      network.append(ReLU())
#      network.append(Dense(j[0], j[1]))
#      network.append(ReLU())
#      network.append(Dense(j[1], j[2]))
#      network.append(ReLU())
#      network.append(Dense(j[2], j[3]))
#      network.append(ReLU())
#      network.append(Dense(j[3], 10))
#      
#      for epoch in range(20):
#
#          for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=128,shuffle=True):
#              train(network,x_batch,y_batch)
#          
#          train_log.append(np.mean(predict(network,X_train)==y_train))
#          val_log.append(np.mean(predict(network,X_val)==y_val))
#          print("Epoch",epoch)
#          print("Train accuracy:",train_log[-1])
#          print("Val accuracy:",val_log[-1])
#          if val_log[-1] > best_accuracy:
#            best_accuracy = val_log[-1]
#            best_num_layers = i
#            best_num_units = j
#
#      plt.plot(train_log,label='train accuracy')
#      plt.plot(val_log,label='val accuracy')
#      plt.legend(loc='best')
#      plt.grid()
#      plt.show()
#
#print("best accuracy: "+str(best_accuracy))
#print("best num layers: "+str(best_num_layers))
#print("best best num units: "+str(best_num_units))

network = []
first_layer = [64,128,256,512]
second_layer = [64,128,256,512]
best_accuracy = 0
best_num_first_layer = 0
best_num_second_layer = 0 

for i in first_layer:
  for j in second_layer:
    network = []
    network.append(Dense(X_train.shape[1], i))
    network.append(ReLU())
    network.append(Dense(i, j))
    network.append(ReLU())
    network.append(Dense(j, 10))
    
    train_log = list()
    val_log = list()
    for epoch in range(20):

        for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=128,shuffle=True):
            train(network,x_batch,y_batch)
        
        train_log.append(np.mean(predict(network,X_train)==y_train))
        val_log.append(np.mean(predict(network,X_val)==y_val))
        print("Epoch",epoch)
        print("Train accuracy:",train_log[-1])
        print("Val accuracy:",val_log[-1])
        if val_log[-1] > best_accuracy:
          best_accuracy = val_log[-1]
          best_num_first_layer = i
          best_num_second_layer = j
    print('number of units in first layer : ' + str(i))
    print('number of units in second layer : ' + str(j))
    file1 = open("Acc.txt","a")
    file1.write('first layer : ' + str(i) + ', second layer : ' + str(j) + ", Acc= " + str(best_accuracy)+ "___\n")
    file1.close() 
    plt.plot(train_log,label='train accuracy')
    plt.plot(val_log,label='val accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
