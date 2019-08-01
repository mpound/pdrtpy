
class ModelRatioData:
    def __init__(self,numerator, denominator, identifier,filename,metallicity=1):
        self._numerator = numerator
        self._denominator =denominator
        self._identifier = identifier
        self._filename = filename + "web"
        self._metallicity= metallicity
        
    @property
    def id(self): return self._identifier

    @property 
    def filename(self): return self._filename
    
    @property 
    def metallicity(self): return self._metallicity
    
    @property
    def isSolarMetallicity(self):
        return self._metallicity == 1

