import time
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

ua = UserAgent(browsers=['Edge', 'Chrome','Safari','Firefox','Google'],platforms=['desktop'],os=['Windows','Linux','Mac OS X'])
HEADER = {"User-Agent": ua.random}

def UpdatedLEGOData(): 
    with open ("URLS.txt", "r") as file:
        URLS_LIST = file.readlines()
        URLS_PY = [URL.strip() for URL in URLS_LIST]
        
    updated_Data = []
    
    for URL_PY in URLS_PY: 
        current_URL_Data = []
        r = requests.get(URL_PY,headers=HEADER)
        htmlSoup = BeautifulSoup(r.content, "html5lib")
        
        product_Id_Tag = htmlSoup.find("meta", property="product:retailer_item_id")
        product_Id = product_Id_Tag["content"] if product_Id_Tag else "Not found product ID"
        
        price_Tag = htmlSoup.find("meta", property="product:price:amount")
        price = price_Tag["content"] if price_Tag else "Not found price"
        
        availability_Tag = htmlSoup.find("meta", property="product:availability")
        availability = availability_Tag["content"] if availability_Tag else "Not found availability"
        
        shipping_Tag = htmlSoup.find("div", class_="ProductOverviewstyles__PriceAvailabilityWrapper-sc-1a1az6h-11 bgwYoN")
        shipping = shipping_Tag.find('span', class_="Markup__StyledMarkup-sc-nc8x20-0 dbPAWk").get_text() 
        if not shipping: 
            shipping = "Not found shipping"
        
        current_URL_Data.append(product_Id)
        current_URL_Data.append(price)
        current_URL_Data.append(availability)
        current_URL_Data.append(shipping)
            
        updated_Data.append(current_URL_Data)
        time.sleep(5)
    return updated_Data
