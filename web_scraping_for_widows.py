import sys
from pathlib import Path
import logging

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import time
import pandas as pd
from datetime import datetime


# Set up basic logging
logging_dir = 'logs/scrape_blockchain.log'
try:
    open(logging_dir, 'r')
except FileNotFoundError:
    print('No file found at {}. Attempting to create new file...' \
            .format(logging_dir))
    open(logging_dir, 'w+')
    print('...success')

logging.basicConfig(
    filename=logging_dir,
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)



def scrape_blockchain(verbose=True):

    # Call the correct webdriver based on operating system
    if sys.platform == 'win32':
        executable_path = './web_drivers/chromedriver.exe'
    elif sys.platform == 'linux':
        executable_path = './web_drivers/chromedriver_linux'
    elif sys.platform in ['darwin', 'os2', 'os2emx']:
        executable_path = './web_drivers/chromedriver_mac'

    # Add arguments telling Selenium to not actually open a window
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--window-size=1920x1080')

    if verbose:
        print(f'Starting browser with chrome driver at {executable_path}')
    # Fire up browser
    browser = webdriver.Chrome(executable_path=executable_path,
                               chrome_options=chrome_options)


    url = 'https://blockchain.info/stats'
    if verbose:
        print(f'Loading {url}')

    # Load webpage
    browser.get(url)
    time.sleep(2) # Pause to make sure that javascript loads

    # The id's we're expecting to find in the HTML


    data_fields = ['market_price_usd','trade_volume_usd','total_btc_sent','n_tx','estimated_btc_sent','difficulty','hash_rate','miners_revenue_usd','miners_cost_per_tx',
    			   'total_fees_btc','minutes_between_blocks','miners_revenue_percent_fees']





    if verbose:
        print('Parsing HTML')
    # Parse HTML, close browser
    soup = BeautifulSoup(browser.page_source, 'html.parser')
    browser.quit()
    text = soup.find_all(id='n_blocks_mined')

    # Get data from HTML, remove symbols, convert to float, store in 'data'
    data = {}
    for field in data_fields:
        text = soup.find(id=field)
        if text:
            text = text.get_text()
            text = ''.join(i for i in text if i.isdigit() or i == '.')
        else:
            logger.warn(f'No data was found for <td id= {field}>')
            text = 'nan'
        data[field] = float(text)

    # Convert data to pandas DataFrame
    data['date'] = datetime.strftime(datetime.now(),'%m-%d-%Y %H:%M:%S')
    data = pd.DataFrame.from_dict(data, orient='index').T
    data.index = data.date
    data = data.drop(['date'], axis=1)

    # Save to CSV. If a file already exists, append to it
    file_location = r'./data/scraped_data.csv'
    p = Path(file_location)
    if p.exists():
        if verbose:
            print(f'Appending data to {file_location}')
        with open(file_location, 'a') as f:
            data.to_csv(f, header=False)
    else:
        if verbose:
            print(f'Saving data to new file at {file_location}')

        data.to_csv(file_location)


if __name__ == '__main__':
    try:
        scrape_blockchain()
    except Exception as err:
        logger.error(err)
