class DecisionTreeClassifier:
    def __init__(self,max_depth):
        self.max_depth = max_depth
        self.tree = {}
        
    def calc_tresholds(self,dframe,obj_name):
        ths = {}
        dframe= dframe.drop(columns=[obj_name])
        for c in dframe.columns:
            values = sorted(dframe[c].unique())
            ths[c] = []
            for v1,v2 in zip(values,values[1:]):
                ths[c].append((v1+v2)/2)
        return ths

    def calc_gini(self,grp,obj_name):
        props = grp[obj_name].value_counts(normalize=True)
        props *= props
        return 1 - sum(props)
        
    def calc_impurity(self,grp1,grp2,obj_name):
        count1 = len(grp1)
        count2 = len(grp2)
        total = count1+count2
        g1 = self.calc_gini(grp1,obj_name)
        g2 = self.calc_gini(grp2,obj_name)
        return (count1/total * g1) + (count2/total * g2)
    
    def calc_best_th(self,dframe,obj_name,ths,verbose=False):
        impurities = {}
        for k,v in ths.items():
            for th in v:
                grp1 = dframe[dframe[k]<=th]
                grp2 = dframe[dframe[k]>th]
                if len(grp1) == 0 or len(grp2) == 0:
                    continue
    
                imp = self.calc_impurity(grp1,grp2,obj_name)
                if verbose:
                    print(f'{k}:{th} --> {imp}')
    
                impurities[(k,th)] = imp
        minkey = min(impurities,key=lambda x:impurities[x])
        return (minkey,impurities[minkey])
        
    def fit(self,dframe,obj_name,current_depth = 0):
        gini = self.calc_gini(dframe,obj_name)
        
        if gini == 0 or current_depth == self.max_depth:
            choice = dframe[obj_name].mode()[0]
            return {'leaf':True,'class':choice}
            
        else:
            ths = self.calc_tresholds(dframe,obj_name)
            best_th = self.calc_best_th(dframe,obj_name,ths)
            col,th = best_th[0]
            grp1 = dframe[dframe[col] <= th]#All the rows where df[th.column]<=th.value
            grp2 = dframe[dframe[col] > th]#Rest of the rows
            
            left = self.fit(grp1,obj_name,current_depth+1) #Fit recursively for left group
            right = self.fit(grp2,obj_name,current_depth+1) #Fit recursively for right group
        
            t = {
                'leaf':False,
                'column':col,
                'th':th,
                'left':left,
                'right':right
            }
            
            if current_depth == 0:
                self.tree = t
                return self.tree
            else:
                return t

    def predict(self,example,node=None):
        if node is None:
            node = self.tree
            
        if node['leaf']:
            return node['class']
        elif example[node['column']] <= node['th']:
            return self.predict(example,node['left'])
        else:
            return self.predict(example,node['right'])
    
    def predict_all(self,df):
        return [self.predict(row) for _,row in df.iterrows()]

    def accuracy(self,obj,preds):
        return 1- len(obj[obj!=preds])/len(obj)