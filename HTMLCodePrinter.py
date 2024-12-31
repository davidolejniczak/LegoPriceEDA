def loadCookies(driver, cookies):
    driver.get('https://www.brickeconomy.com')  # Load the base page
    for cookie in cookies:
        driver.add_cookie(cookie)
    driver.refresh()

def getPageSource(url,driver):
    try:
        cookies = cookieMaker(driver)
        if not cookies:
            print("No cookies found in getPageSource from cookieMaker")
        loadCookies(driver,cookies)
        driver.get(url)
        HTMLCode = driver.page_source
        if not HTMLCode:
            print("No HTML Code found in getPageSource")
        return HTMLCode
    except Exception as e:
        print(f'An error occurred: {e}')

def cookieMaker(driver):
    try:
        url = 'https://www.brickeconomy.com'
        driver.get(url)
        ''' input("Solve CAPTCHA if presented, then press Enter...")
        #why do I have to press something sometimes for it to work
        #maybe do a auto button click '''

        cookies = driver.get_cookies()
        return cookies

    except Exception as e:
        print(f"An error occurred: {e}")