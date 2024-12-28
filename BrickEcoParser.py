from bs4 import BeautifulSoup

def parseBrickEco():
    filePath = 'page_source.html'
    with open(filePath, 'r', encoding='utf-8') as file:
        HtmlCode = file.read()
        
    soup = BeautifulSoup(HtmlCode, 'html.parser')

    allRowlits = soup.find_all('div', class_='row rowlist')

    RetiredDate = ''
    MarketPriceStillAvailable = ''
    MarketPriceRetired = ''
    Theme = '' 
    RetirementPredictionPop = ''
    RetirementDataList = []
    RetirementData = ''

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
            
    themeRows = soup.find_all('div', class_='semibold bdr-b-l pb-2 mb-6')         
    for themeRow in themeRows:
        Theme = themeRow.get_text(separator=' ', strip=True)

    MarketPriceStillAvailable = MarketPriceStillAvailable[:-6]+' '+MarketPriceStillAvailable[-6:]
    RetirementData = RetirementDataList[0]
    RetirementData = RetirementData[:-5]+' '+RetirementData[-5:]

    print(RetirementData)
    print(RetirementPredictionPop)
    print(Theme)
    print(MarketPriceStillAvailable)
    print(RetiredDate)
