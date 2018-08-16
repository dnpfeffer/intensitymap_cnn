import wget
import requests
from bs4 import BeautifulSoup
from os.path import isfile

### location to save things to
saveLocation = '../catalogues/'

### url to get catalogs from
url = 'http://www.cita.utoronto.ca/~gstein/data/CO/COMAP_fullvolume/peak-patch-runs/'

### get all of the links on the webpage
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')
links = soup.find_all('a')

### get the links to .npz files, save them and just the file names
goodLinks = []
for link in links:
    if '.npz' not in link.contents[0]:
        continue
    goodLinks.append(link.contents[0])

print('There are {} catalogs to download'.format(len(goodLinks)))
for i, link in enumerate(goodLinks):
    print('\nDownloading file number {}'.format(i))
    if isfile(saveLocation + link):
        continue
    wget.download(url + link, saveLocation + link)






