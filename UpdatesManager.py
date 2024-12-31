import HTMLCodePrinter as HP
import BrickEcoParser as BEP
import os

def BrickEconomyData():
    with open ("BRICK_ECONOMY_URLS.txt", "r") as file:
        BrickUrlsList = file.readlines()
        BrickUrlsPY = [UrlBrick.strip() for UrlBrick in BrickUrlsList]
    updatedBrickData = []
    for BrickUrl in BrickUrlsList:
        HP.getPageSource(BrickUrl)
        updatedBrickData.append(BEP.parseBrickEco())
         
        os.remove("page_source.html")

def main ():
    BrickEconomyData()
    
if __name__ == "__main__": 
    main()