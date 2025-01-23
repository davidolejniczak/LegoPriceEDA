import xgboost as xgb
import pandas as pd

def importModel(): 
    xgbModel = xgb.Booster()
    xgbModel.load_model(xgb_lego_model.json)


def main():
    importModel()
    
if __name__=="__main__":
    main()