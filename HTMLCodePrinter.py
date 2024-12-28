from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from fake_useragent import UserAgent

ua = UserAgent(browsers=['Chrome','Google'],platforms=['desktop'],os=['Windows','Linux','Mac OS X'],)

chromeOptions = Options()
chromeOptions.add_argument('--headless')
chromeOptions.add_argument('--disable-gpu')
chromeOptions.add_argument('--no-sandbox')
chromeOptions.add_argument('--disable-dev-shm-usage')
chromeOptions.add_argument(ua.random)

driver = webdriver.Chrome(service=Service('/usr/bin/chromedriver'), options=chromeOptions)

def getPageSource(url):
    try:
        url = url
        driver.get(url)
        wait = WebDriverWait(driver,20)
        with open('page_source.html', 'w') as file:
            file.write(driver.page_source)
    
    except Exception as e:
        print(f'An error occurred: {e}')

    finally:
        driver.quit()
