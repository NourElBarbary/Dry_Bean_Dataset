import numpy as np
class Neuron():
    def __init__(self,random_state=42,lr=0.001,activation="sigmoid",biasFlag=True,input=None,output_Neuron=False,Y_Actual=None) -> None:
        np.random.seed(random_state)
        
        self.input = input
        self.activation=activation
        self.learning_rate=lr
        self.weights=None
        self.biasFlag=biasFlag
        self.bias=None
        self.output=0
        self.error=None
        
        self.Y_Actual=Y_Actual
        
        
    
    def init_weights(self,input_size):

        self.weights=np.random.randn(input_size,1)

        if self.biasFlag:
            self.bias = 1
        else:
            self.bias=0

    #Forward Functions
    def linear_forward(self):
        Z= np.dot(self.weights.T,self.input.T)+self.bias
        return Z
    
    def sigmoid(self,Z):
        return 1 / (1 + np.exp(-Z))
    
    def tanh(self,Z):
        return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    
    def activation_forward(self):
        Z = self.linear_forward()
        if self.activation=='sigmoid':

            A = self.sigmoid(Z)

        elif self.activation=='tanh':

            A = self.tanh(Z)
        
        self.output=A

        return A
    
    #Backward Functions


    def sigmoid_backward(self):
        return self.output * (1-self.output)
    
    def tanh_backward(self):
        return 1 - np.power(self.output,2)
    
    def NeuronError(self,NextError=None,OutputLayer=False,weights=None): ## error of next layer
        if self.activation=='sigmoid':

            Drev = self.sigmoid_backward()

        elif self.activation=='tanh':

            Drev = self.tanh_backward()

        if OutputLayer:


            self.error=(self.Y_Actual-self.output)*Drev
            return self.error
        else:
            
            # print(NextError)
            # print(weights.T)
            # print(np.dot(NextError,weights.T))
            self.error=Drev*np.sum(np.dot(NextError,weights.T))
            return self.error
        
        # print(self.error.shape)
    def update_weights(self):
        # print(self.weights)
        # print("Neuron error",self.error.shape)
        # print("Neuron Inputs",self.input.shape)
        # print("Neuron Weights Before Update",self.weights.shape)
        self.weights=self.weights+(self.learning_rate*np.dot(self.error,self.input).T)
        # print("Neuron Weights After",self.weights.shape)

class Layer():
    def __init__(self,random_state=42,lr=0.001,activation="sigmoid",biasFlag=True,layer_input=None,neurons=1) -> None:
        np.random.seed(random_state)
        self.weights=[i for i in range(neurons)]
        self.biasFlag=biasFlag
        self.learning_rate=lr
        self.activation=activation
        self.neurons=[Neuron(random_state=random_state+i,lr=lr,activation=activation,biasFlag=biasFlag,input=layer_input) for i in range(neurons)]
        self.layer_input=layer_input
        self.outputlayer=False
        self.layerOutput=[i for i in range(len(self.neurons))]
        self.layerError=[i for i in range(len(self.neurons))]
    
    def init_weights(self,input_size):
        
        self.weights=[i for i in range(len(self.neurons))]
        for i in range(len(self.neurons)):
            self.neurons[i].init_weights(input_size)
            self.weights[i]=self.neurons[i].weights
        self.weights=np.array(self.weights).reshape(-1,len(self.neurons))
        
    
    #Forward Functions
    
    def forward(self,X=None):
        self.layer_input=X
        self.layerOutput=[i for i in range(len(self.neurons))]
        for i in range(len(self.neurons)):
            self.neurons[i].input=X
            self.layerOutput[i]=self.neurons[i].activation_forward()
        self.layerOutput=np.array(self.layerOutput).reshape(-1,len(self.neurons))
    
    def backward(self,Y_actual=None,error=None,weights=None):
        
        self.layerError=[i for i in range(len(self.neurons))]
        
        for i in range(len(self.neurons)):
            self.neurons[i].Y_Actual=Y_actual
            self.layerError[i]=self.neurons[i].NeuronError(error,self.outputlayer,weights)

        self.layerError=np.array(self.layerError).reshape(-1,len(self.neurons))
        # print("Layer error",self.layerError.shape)
    
    def update_weights(self):
        
        self.weights=[i for i in range(len(self.neurons))]
        
        for i in range(len(self.neurons)):
            
            self.layerError[i]=self.neurons[i].update_weights()
            self.weights[i]=self.neurons[i].weights
        self.weights=np.array(self.weights).reshape(-1,len(self.neurons))

class Network():

    def __init__(self,num_layers:int,neurons:list,random_state=42,lr=0.001,activation="sigmoid",bias=True,epochs=10) -> None:
        # np.random.seed(random_state)
        if(len(neurons)!=num_layers):
            print("Number of neurons per layer do not match number of layers")
            return
        self.result=None
        self.num_layers=num_layers+1 #output layer
        neurons.append(3) # output layer
        self.layers=[Layer(random_state=random_state,lr=lr,activation=activation,biasFlag=bias,neurons=neurons[i]) for i in range(self.num_layers)]
        self.epochs=epochs
        


    def Train(self,X,y):

        #Initialization
        self.layers[0].init_weights(X.shape[1])
        # print("Layer # 1\n",self.layers[0].weights)

        for i in range(1,self.num_layers):
            self.layers[i].init_weights(len(self.layers[i-1].layerOutput))
            # print(f"Layer # {i+1}\n",self.layers[i].weights)
        
        self.layers[-1].outputlayer=True
        
        #--------------------------------------------------------
    
        for _ in range(self.epochs):

        #Forward
            
            self.layers[0].forward(X) #(5,120)
            
            for i in range(1,self.num_layers):
                
                self.layers[i].forward(self.layers[i-1].layerOutput)
                # print(self.layers[i].layer_input.shape)
                # print(self.layers[i].layerOutput.shape)
            
            self.result=self.layers[-1].layerOutput
        
        #Backward Propagation

            self.layers[-1].backward(np.argmax(y,axis=1))
            
            for i in reversed(range(0,self.num_layers-1)):
                self.layers[i].backward(Y_actual=None,error=self.layers[i+1].layerError,weights=self.layers[i+1].weights)
        
        # Update
            for i in range(self.num_layers):
                self.layers[i].update_weights()

        
    def Test(self,X):
        

        self.layers[0].forward(X) #(5,120)
            
        for i in range(1,self.num_layers):
                
            self.layers[i].forward(self.layers[i-1].layerOutput)
            # print(self.layers[i].layer_input.shape)
            # print(self.layers[i].layerOutput.shape)
        
        self.result=self.layers[-1].layerOutput
        
        return np.argmax(self.layers[-1].layerOutput,axis=1)
    