import time
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

ua = UserAgent(browsers=['Edge', 'Chrome','Safari','Firefox','Google'],platforms=['desktop'],os=['Windows','Linux','Mac OS X'],)
HEADER = {"User-Agent": ua.random}

def UpdatedLEGOData(): 
    with open ("LEGOURLS.txt", "r") as file:
        UrlLegoList = file.readlines()
        UrlSLegoPY = [UrlLego.strip() for UrlLego in UrlLegoList]
        
    updated_Data = []
    
    for LegoUrl in UrlSLegoPY: 
        current_URL_Data = []
        r = requests.get(LegoUrl,headers=HEADER)
        htmlSoup = BeautifulSoup(r.content, "html5lib")
        
        productIdTag = htmlSoup.find("meta", property="product:retailer_item_id")
        productId = productIdTag["content"] if productIdTag else "Not found product ID"
        
        priceTag = htmlSoup.find("meta", property="product:price:amount")
        price = priceTag["content"] if priceTag else "Not found price"
        
        availabilityTag = htmlSoup.find("meta", property="product:availability")
        availability = availabilityTag["content"] if availabilityTag else "Not found availability"
        
        shipping_Tag = htmlSoup.find("div", class_="ProductOverviewstyles__PriceAvailabilityWrapper-sc-1a1az6h-11 bgwYoN")
        shipping = shipping_Tag.find('span', class_="Markup__StyledMarkup-sc-nc8x20-0 dbPAWk").get_text() 
        if not shipping:
            shipping = "Not found shipping"
        
        current_URL_Data.append(productId)
        current_URL_Data.append(price)
        current_URL_Data.append(availability)
        current_URL_Data.append(shipping)
            
        updated_Data.append(current_URL_Data)
        time.sleep(5)
    return updated_Data