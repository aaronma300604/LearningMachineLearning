import pandas as pd
import numpy as np
from collections import Counter

class KnnClassifier:
    def __init__(self,X,y,k,metric = 'euclidean'):
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        self.k = k
        self.metric = metric

    def calc_euclidean(self,e1,e2):
        res = np.sum(np.square(e2-e1))
     
        return np.sqrt(res)

    def calc_manhattan(self,e1,e2):
        res = np.sum(np.abs(e2-e1))
        return res
        
    
    def predict(self,example):
        example = example.to_numpy() if hasattr(example,'to_numpy') else example
        distances = []
        for i in range(len(self.X)):
            ex = self.X[i]
            if self.metric == 'manhattan':
                distances.append((i,self.calc_manhattan(ex,example)))
            if self.metric == 'euclidean':
                distances.append((i,self.calc_euclidean(ex,example)))
        distances = sorted(distances,key = lambda x: x[1])[:self.k]
        neighbor_values = self.y[list(map(lambda x: x[0],distances))]
        return Counter(neighbor_values).most_common(1)[0][0]

    def predict_all(self,df):
        preds = []
        for i in range(len(df)):
            ex = df.iloc[i].to_numpy()
            preds.append(self.predict(ex))
        return preds
