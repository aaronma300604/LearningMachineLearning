import pandas as pd
import numpy as np

class NaiveBayesClassifier:
    def __init__(self,k):
        self.k = k
        self.probabilities = {}
    
    def fit(self,dataf:pd.DataFrame, x_names:list, y_names:list):
        groups = dataf.groupby(y_names)
        self.classes = list(groups.indices.keys())
        self.numClasses = len(self.classes)
        self.possible_values = {a: dataf[a].unique() for a in x_names}
        
        for c in self.classes:
                            # Condition prevents warnings when classes are (not) tuples
            g = groups.get_group(c if type(c) == tuple else (c,)) #For each class we calc
            self.probabilities[c] = {}
            self.probabilities[c][c] = int(groups.size()[c])/sum(groups.size()) #Probability of that class
            for a in x_names:
                self.probabilities[c][a] = (g[a].value_counts().reindex(self.possible_values[a],fill_value=0) + self.k) \
                    /(g.size +self.k*len(self.possible_values[a]))
                    #Conditioned probabilities for that class and each attribute value

    def predict(self,example:pd.Series):
            preds = {}
            for c in list(self.probabilities.keys()):
                pred = np.log(self.probabilities[c][c]) #Probability of a class. We use logs to prevent underflow[log(a)+log(b)=log(ab)]
                for a in list(example.keys()):
                    pred += np.log(self.probabilities[c][a][example[a]]) #Conditional probability of tha atribute value for that class
                preds[c] = float(pred)
            return preds,max(preds,key=lambda x: preds[x])

    def predict_all(self, examples:pd.DataFrame):
        preds = np.zeros((len(examples),),dtype=object)
        for i in range(len(examples)):
            preds[i] = self.predict(examples.iloc[i])[1]
        return preds