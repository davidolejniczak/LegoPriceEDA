import HTMLCodePrinter as HP

def BrickEconomyData():
    with open ("BRICK_ECONOMY_URLS.txt", "r") as file:
        BrickUrlsList = file.readlines()
        BrickUrlsPY = [UrlBrick.strip() for UrlBrick in BrickUrlsList]
    updatedBrickData = []
    for BrickUrl in BrickUrlsList:
        HP.getPageSource(BrickUrl)

        