import numpy as np
from Model import Model

class Adaline(Model):
    def __init__(self,random_state=42,epochs=1000,lr=0.001,bias=True) -> None:
        np.random.seed(random_state)
        super().__init__(random_state,epochs=epochs,lr=lr,bias=bias)
        


    def fit(self,X,Y,MSE_Threshold):
        if self.bias:
            X=np.concatenate([np.ones((X.shape[0],1)),X.to_numpy()],axis=1)
        else:
            X=X.to_numpy()
        self.weight=np.random.rand(X.shape[1]).reshape(-1,1)
        print(self.weight)
        for _ in range(self.epochs):
            for i,x in enumerate(X):

                self.net_value = np.dot(self.weight.T,x.reshape(-1,1))

                error = Y.iloc[i]-self.net_value[0][0]

                self.weight = self.weight - self.learning_rate * error * x.reshape(-1,1)
                
            for i,x in enumerate(X):
                 self.net_value = np.dot(self.weight.T,x.reshape(-1,1))
                 MSE=0
                 MSE+=0.5*((Y.iloc[i]-self.net_value)**2)
            if (MSE/len(X))<MSE_Threshold:
                break


    def predict(self,X_test):
        if self.bias:
            X_test=np.concatenate([np.ones((X_test.shape[0],1)),X_test],axis=1)
        else:
            X_test=X_test.to_numpy()
        self.net_value = np.dot(self.weight.T,X_test.T)

        return np.where(self.net_value >= 0, 1, -1)
    
