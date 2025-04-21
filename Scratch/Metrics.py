def accuracy(obj,preds):
        return 1- len(obj[obj!=preds])/len(obj)