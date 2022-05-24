class Scaler(object):
    """ scales data before feeding to classifier"""
    def __init__(self, classifier , scaler)     :
        super(Scaler, self).__init__()
        self.classifier = classifier
        self.scaler = scaler

    def predict(self , X):
        X = self.scaler.transform(X)
        return self.classifier.predict(X)
    def predict_proba(self , X) :
        X = self.scaler.transform(X)
        return self.classifier.predict_proba(X)

    def  get_classes(self):
        return self.classifier.classes_ 
