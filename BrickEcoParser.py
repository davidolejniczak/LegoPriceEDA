from bs4 import BeautifulSoup as BS

def parseBrickEco(HTMLCode):
    MarketPriceStillAvailable = ''
    MarketPriceRetired = ''
    RetirementDataList = []
    RetirementData = ''
    RetiredDate = ''
    RetirementPredictionPop = ''
    Theme = '' 
    allData = []

    if not HTMLCode:
        print("No HTML Code found in BrickEcoParser")
        
    soup = BS(HTMLCode, 'html.parser')
    allRowlits = soup.find_all('div', class_='row rowlist')
    
    for row in allRowlits:
        label = row.find('div', class_='col-xs-5 text-muted')
        value = row.find('div', class_='col-xs-7')
        if label and 'Market price' in label.text:
            MarketPriceStillAvailable = value.text.strip()
        if label and value and 'Retired' in label.get_text(strip=True):
            RetiredDate = (value.get_text(strip=True))
        if label and value and 'Retirement' in label.text:
            RetirementPredictionPop = (value.get_text(strip=True))
        if label and 'Retirement' in label.get_text(strip=True):
            RetirementText = value.get_text(strip=True)
            RetirementDataList.append(RetirementText)
        if label and 'Value' in row.get_text():
            value_div = row.find('div', class_='col-xs-7')
            if value_div and value_div.find('b'):
                MarketPriceRetired = value_div.find('b').text
    
    themeRows = soup.find_all('div', class_='semibold bdr-b-l pb-2 mb-6')         
    for themeRow in themeRows:
        Theme = themeRow.get_text(separator=' ', strip=True)

    try: 
        MarketPriceStillAvailable = MarketPriceStillAvailable[:-6]+' '+MarketPriceStillAvailable[-6:]
    except: 
        pass
    
    try: 
        RetirementData = RetirementDataList[0]
    except: 
        pass
    
    try: 
        RetirementData = RetirementData[:-5]+' '+RetirementData[-5:]
    except:
        pass
    
    MarketPrice = ''
    if not MarketPriceRetired:
        MarketPrice = MarketPriceStillAvailable
    else:
        MarketPrice = MarketPriceRetired
    
    RetDDate = ''
    if not RetiredDate:
        RetDDate = RetirementData
    else: 
        RetDDate = RetiredDate
    
    if not RetirementPredictionPop:
        RetirementPredictionPop = 'Already Retired'
    
    allData.append(MarketPrice)
    allData.append(RetDDate)
    allData.append(RetirementPredictionPop)
    allData.append(Theme)
    #print(allData)
    return allData

'''def main():
    parseBrickEco()
    
if __name__ == "__main__":
    main()'''