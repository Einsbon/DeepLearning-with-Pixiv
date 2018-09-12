from selenium import webdriver
from bs4 import BeautifulSoup
import time
import urllib.request
import urllib
import getpass
import os
import shutil

phantompath = r'C:\Users\Einsbon\Documents\Python_Projects\python_window\crawler\pixiv_low_quality_image_crawler\phantomjs.exe'
chromepath = 'C:\\Users\\Einsbon\\Documents\\Python Scripts\\crawler\\pixiv_low_quality_image_crawler\\chromedriver.exe'
urlVocaloid = r"https://www.pixiv.net/search.php?s_mode=s_tag_full&word=VOCALOID&type=illust&blt=100&mode=safe"
urlMiku = r'https://www.pixiv.net/search.php?s_mode=s_tag_full&word=%E5%88%9D%E9%9F%B3%E3%83%9F%E3%82%AF&type=illust&blt=100&mode=safe'
urlLogin = 'https://accounts.pixiv.net/login?lang=ko&source=pc&view_type=page&ref=wwwtop_accounts_index'


def driverSetup(web, userId, userpd, urlFirst):
    web.get(urlFirst)
    web.implicitly_wait(10)
    web.find_element_by_xpath(
        '/html/body/div[3]/div[3]/div/form/div[1]/div[1]/input').send_keys(
        userId)
    web.find_element_by_xpath(
        '/html/body/div[3]/div[3]/div/form/div[1]/div[2]/input').send_keys(
        userpd)
    web.find_element_by_xpath(
        '/html/body/div[3]/div[3]/div/form/button').click()
    web.set_window_size(1020, 960)


def scrollUpToDown(web):
    web.execute_script("window.scrollTo(0, 400);")
    time.sleep(0.15)
    web.execute_script("window.scrollTo(0, 800);")
    time.sleep(0.15)
    web.execute_script("window.scrollTo(0, 1200);")
    time.sleep(0.15)
    web.execute_script("window.scrollTo(0, 1600);")
    time.sleep(0.15)
    web.execute_script("window.scrollTo(0, 2000);")
    time.sleep(0.15)
    web.execute_script("window.scrollTo(0, 2400);")
    time.sleep(0.15)
    web.execute_script("window.scrollTo(0, 2800);")
    time.sleep(0.15)
    web.execute_script("window.scrollTo(0, 3200);")


def checkLoaded(elements):
    for element in elements:
        est = str(element)
        if est.find('p0_master1200') == -1:
            print(est)
            return False
    return True


opener = urllib.request.URLopener()
opener.addheader('Referer', 'https://www.pixiv.net/')


def downloadImage(url, name):
    image = opener.open(url)
    data = image.read()
    f = open(name, 'wb')
    f.write(data)
    f.close()
    image.close()
    '''

    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
               'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
               'Accept-Encoding': 'none',
               'Accept-Language': 'en-US,en;q=0.8',
               'Connection': 'keep-alive'}
    req = urllib.request.Request(url, headers=headers)  # The assembled request
    response = urllib.request.urlopen(req)  # store the response
    # create a new file and write the image
    f = open(name, 'wb')
    f.write(response.read())
    f.close()'''


def crawling(web, repeatNum, savePath):
    while repeatNum > 0:
        print('\n\nNumber of pages remaining: ' + str(repeatNum))
        print('Current_url:' + str(web.current_url))
        scrollUpToDown(web)
        web.implicitly_wait(20)

        html = web.page_source
        soup = BeautifulSoup(html, 'html.parser')

        count = 0
        elements = soup.find_all('div', {'class': '_309ad3C'})
        print(checkLoaded(elements))
        while checkLoaded(elements) == False:
            print('waiting')
            time.sleep(0.8)
            elements = soup.find_all('div', {'class': '_309ad3C'})
            if count == 5:
                scrollUpToDown(web)
            if count > 12:
                web.refresh()
                scrollUpToDown(web)
                count = 0
            count += 1

        urlList = []
        downloadedWell = True
        downloadcount = 0
        for element in elements:
            # print(element)
            est = str(element)
            if (est.find('(') != -1 & est.find(')') != -1):
                est = est[est.find('(')+2: est.find(')')-1]
                urlList.append(est)
                filename = savePath + "\\" + est.split('/')[-1]
                print(est)
                print(filename)
                try:
                    if os.path.isfile(filename):
                        print(': already')
                    else:
                        downloadImage(est, filename)
                        print(': downloaded')
                        downloadcount += 1
                except:
                    print('error, refrestart this page')
                    downloadedWell = False
                    break

        if downloadedWell == False:
            web.refresh()
            continue
        print('Downloaded images in this page: ' + str(downloadcount))
        web.find_element_by_xpath(
            '//*[@id="wrapper"]/div[1]/div/nav/div/span[2]/a').click()
        repeatNum -= 1


def main():
    # print("launch path" + os.getcwd())
    # print('phantom path' + phantompath)
    print(' ')
    #userId = input('Id:')
    #userPd = getpass.getpass('Password:')
    userId = 'sbkim0316@naver.com'
    userPd = 'airplane0316'
    web = webdriver.Chrome(os.path.abspath(os.path.dirname(__file__)) + '/chromedriver.exe')
    driverSetup(web, userId, userPd, urlLogin)
    #startUrl = input('Start url:')
    startUrl = r'https://www.pixiv.net/search.php?s_mode=s_tag_full&word=VOCALOID&type=illust&blt=1000&mode=safe'
    #savePath = input('Path to save:')
    savePath = r'D:\picture\miku300-999'
    pageNumber = int(input('Number of pages to download:'))
    web.get(startUrl)
    crawling(web, pageNumber, savePath)


if __name__ == "__main__":
    main()
