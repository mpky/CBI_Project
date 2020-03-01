import requests
from bs4 import BeautifulSoup
import time
import os
import numpy as np
import pandas as pd


partial_url = 'https://cbi.iq/currency_auction/view/'
ua1 = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) " \
      "Chrome/66.0.3359.139 Safari/537.36"
headers = {"User-Agent": ua1}


def scraper_124_394():
    df = pd.DataFrame(columns=['date', 'total_for_foreign',
                               'total_cash', 'grand_total', 'exchange_rate'])
    for page in range(124, 395):

        # piece together each unique URL
        whole_url = partial_url + str(page)
        print(whole_url)
        try:
            page = requests.get(whole_url, headers=headers, timeout=500)
            soup = BeautifulSoup(page.content, "html.parser")

            info = soup.find_all(
                'span', attrs={"style": "font-family:Arial,Helvetica,sans-serif"})
            # put data into a list
            data = [i.text for i in info]
            df = df.append({'date': data[1],
                            'total_for_foreign': data[5],
                            'total_cash': data[7],
                            'grand_total': data[9],
                            'exchange_rate': data[-3],
                            'url':whole_url}, ignore_index=True)
        except Exception:
            print('No URL at that number')

        time.sleep(1.2)
    df.to_csv('../data/raw/124_394.csv',index=False)

scraper_124_394()
print('DONE')
