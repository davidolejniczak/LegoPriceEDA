import xgboost as xgb
import pandas as pd

class SetInfo:
    def __init__(self,LegoLink,BrickEcoLink,Name): 
        self.LegoLink = LegoLink
        self.BrickEcoLink = BrickEcoLink
        self.Name = Name
        self.Stats = None
    
    def setStats(self,stats):
        self.Stats = stats
        
        
class SetData:
    def __init__(self,numParts,numFigs,rPrice,numUniqueFigs,setRating,popPred):
        self.NumParts = numParts
        self.NumFigs = numFigs
        self.rPrice = rPrice
        self.numUniqueFigs = numUniqueFigs
        self.setRating =setRating
        self.popPred = popPred
        
def importModel(): 
    xgbModel = xgb.Booster()
    xgbModel.load_model(xgb_lego_model.json)

def main():
    importModel()
    
if __name__=="__main__":
    main()