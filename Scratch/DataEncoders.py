class CategoricalEncoder:
    def __init__(self):
        self.encodings={}
        self.array= None
        self.atrb_names = None
        self.df = None

        
    def fit(self,array):
        self.array=array
        classes = sorted(self.array.unique())
        for i in range(len(classes)):
            self.encodings[classes[i]] = i;
            
    def transform(self):
        res = np.empty(self.array.shape,dtype=float)
        for i in range(len(self.array)):
            res[i] = self.encodings[self.array[i]]
        return res
    
    def fit_df(self,df,atrb_names):
        self.atrb_names =atrb_names
        self.df = df
        for col in self.df.columns:
            if col in self.atrb_names:
                classes = sorted(df[col].unique())
                self.encodings[col] = {cl:i for i,cl in enumerate(classes)}
            
    def transform_df(self):
        res = self.df.copy()
        for col in self.atrb_names:
            res[col] = res[col].map(self.encodings[col])
        return res
    
    def fit_transform_df(self,df,atrb_names):
        self.fit_df(df,atrb_names)
        return self.transform_df()
    
    def fit_transform(self,array):
        self.fit(array)
        return self.transform()
    
    def inverse_transform(self,encoded_array):
        res = np.empty(encoded_array.shape,dtype=object)
        for i in range(len(encoded_array)):
            res[i] = self.array[encoded_array[i]]
        return res
    
    def inverse_transform_df(self,encoded_df):
        res = encoded_df.copy()
        for col in self.atrb_names:
            res[col] = res[col].map({v:k for k,v in self.encodings[col].items()})
        return res
