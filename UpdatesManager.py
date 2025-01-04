import HTMLCodePrinter as HP
import BrickEcoParser as BEP
import SheetUpdater as SU
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
import time

ua = UserAgent(browsers=['Chrome','Google'],platforms=['desktop'],os=['Windows','Linux','Mac OS X'],)

chromeOptions = Options()
chromeOptions.add_argument('--disable-gpu')
chromeOptions.add_argument('--no-sandbox')
chromeOptions.add_argument('--disable-dev-shm-usage')
chromeOptions.add_argument('--remote-debugging-port=9222')
chromeOptions.add_argument(ua.random)

driver = webdriver.Chrome(service=Service('/usr/bin/chromedriver'), options=chromeOptions)

def BrickEconomyData():
    with open ("URLs/BRICK_ECONOMY_URLS.txt", "r") as file:
        BrickUrlsList = file.readlines()
        BrickUrlsPY = [UrlBrick.strip() for UrlBrick in BrickUrlsList]
    updatedBrickData = []
    for BrickUrl in BrickUrlsPY:
        HTMLCode = HP.getPageSource(BrickUrl,driver)
        updatedBrickData.append(BEP.parseBrickEco(HTMLCode))
        time.sleep(5)
    SU.brickUpdateValues(updatedBrickData)
    driver.quit()
    print(updatedBrickData)
def main ():
    BrickEconomyData()
    
if __name__ == "__main__": 
    main()