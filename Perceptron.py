import numpy as np
from Model import Model

class SLP(Model):
    def __init__(self,random_state=42,epochs=1000,lr=0.001,bias=True) -> None:
        np.random.seed(random_state)
        super().__init__(random_state,epochs=epochs,lr=lr,bias=bias)

    def fit(self,X,Y):
        if self.bias:
            X=np.concatenate([np.ones((X.shape[0],1)),X.to_numpy()],axis=1)
        else:
            X=X.to_numpy()
        self.weight=np.random.rand(X.shape[1]).reshape(-1,1)
        # print("Initial Weights:" , self.weight)
        for _ in range(self.epochs):
            for i,x in enumerate(X):
                

                self.net_value = np.dot(self.weight.T,x.reshape(-1,1))
                # print(self.net_value[0][0])
                if self.net_value[0][0]>0:
                    self.A = 1
                elif self.net_value[0][0]<0:
                    self.A = -1
                else:
                    self.A = 0

                if Y.iloc[i]!=self.A:
              
                    error = Y.iloc[i]-self.A
                    self.weight=self.weight + self.learning_rate*(error*x.reshape(-1,1))


    def predict(self,X_test):
        if self.bias:
            X_test=np.concatenate([np.ones((X_test.shape[0],1)),X_test],axis=1)
        else:
            X_test=X_test.to_numpy()
        self.net_value = np.dot(self.weight.T,X_test.T)
        self.net_value=np.where(self.net_value>0,1,np.where(self.net_value<0,-1,0))
        return self.net_value