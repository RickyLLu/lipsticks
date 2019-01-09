# -*- coding:utf-8 -*-
import bs4
import urllib2
import re
import os
import threading
import Queue
import time
import chardet
import jieba

import selenium
from selenium import webdriver
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def h0(key):
    hash = 0xAAAAAAAA
    for i in range(len(key)):
      if ((i & 1) == 0):
        hash ^= ((hash << 7) ^ ord(key[i]) * (hash >> 3))
      else:
        hash ^= (~((hash << 11) + ord(key[i]) ^ (hash >> 5)))
    return hash


def h1(key):
    a = 378551
    b = 63689
    hash = 0
    for i in range(len(key)):
      hash = hash * a + ord(key[i])
      a = a * b
    return hash


def h2(key):
    hash = 1315423911
    for i in range(len(key)):
      hash ^= ((hash << 5) + ord(key[i]) + (hash >> 2))
    return hash


def h3(key):
   BitsInUnsignedInt = 4 * 8
   ThreeQuarters     = long((BitsInUnsignedInt  * 3) / 4)
   OneEighth         = long(BitsInUnsignedInt / 8)
   HighBits          = (0xFFFFFFFF) << (BitsInUnsignedInt - OneEighth)
   hash              = 0
   test              = 0

   for i in range(len(key)):
     hash = (hash << OneEighth) + ord(key[i])
     test = hash & HighBits
     if test != 0:
       hash = (( hash ^ (test >> ThreeQuarters)) & (~HighBits));
   return (hash & 0x7FFFFFFF)


def h4(key):
    hash = 0
    x    = 0
    for i in range(len(key)):
      hash = (hash << 4) + ord(key[i])
      x = hash & 0xF0000000
      if x != 0:
        hash ^= (x >> 24)
      hash &= ~x
    return hash


def h5(key):
    seed = 131 # 31 131 1313 13131 131313 etc..
    hash = 0
    for i in range(len(key)):
      hash = (hash * seed) + ord(key[i])
    return hash


def h6(key):
    hash = 0
    for i in range(len(key)):
      hash = ord(key[i]) + (hash << 6) + (hash << 16) - hash;
    return hash


def h7(key):
    hash = 5381
    for i in range(len(key)):
       hash = ((hash << 5) + hash) + ord(key[i])
    return hash


def h8(key):
    hash = len(key);
    for i in range(len(key)):
      hash = ((hash << 5) ^ (hash >> 27)) ^ ord(key[i])
    return hash


def h9(key):
    hash = 0
    for i in range(len(key)):
       hash = hash << 7 ^ ord(key[i])
    return hash


def valid_filename(s):
    import string
    valid_chars = "-_()=:&? %s%s" % (string.ascii_letters, string.digits)
    s = ''.join(c for c in s if c in valid_chars)
    return s


def get_page(page):
    try:
        with ioLock:
            print 'Downloading page %s' % page
            time.sleep(0.1)
            head = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.79 Safari/537.36'
            userAgent = {'User-Agent': head}
            request = urllib2.Request(page, headers=userAgent)
            content = urllib2.urlopen(request, timeout=5).read()
            type = chardet.detect(content[:400])['encoding']
            content = content.decode(type, 'ignore')

            code = re.findall(r'\d+', page, re.S)
            code = code[0]
        return content, code
    except:
        return "ERROR"


'''获取所有URL所用的getpage'''
def get_Frontpage(page):
    try:
        with ioLock:
            print 'Downloading page %s' % page
            time.sleep(0.1)
            head = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.79 Safari/537.36'
            userAgent = {'User-Agent': head}
            request = urllib2.Request(page, headers=userAgent)
            content = urllib2.urlopen(request, timeout=10)
            content = content.read()
            type = chardet.detect(content[:400])['encoding']
            content = content.decode(type, 'ignore')
        return content
    except:
        return "ERROR"


def getPrice(code):
    import json
    url = 'https://p.3.cn/prices/mgets?skuIds=J_' + code
    # 获取地址
    request = urllib2.Request(url)
    # 打开连接
    response = urllib2.urlopen(request)
    content = response.read()
    result = json.loads(content)
    try:
        json = result[0]
        return json['p']
    except:
        return 0


def getRate(code):
    import json
    res = []
    generalCommentDic = {}

    url = "http://club.jd.com/productpage/p-" + str(code) + "-s-0-t-3-p-0.html"
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    content = response.read()
    try:
        content = content.decode('gbk')
    except:
        pass
    result = json.loads(content)

    comments = result["productCommentSummary"]

    try:
        '''总评价数'''
        commentCount = comments["commentCount"]
        '''好评度'''
        goodRateShow = comments["goodRateShow"]
        '''中评度'''
        generalRateShow = comments["generalRateShow"]
        '''差评度'''
        poorRateShow = comments["poorRateShow"]
        '''好评数'''
        goodCount = comments["goodCount"]
        '''中评数'''
        generalCount = comments["generalCount"]
        '''差评数'''
        poorCount = comments["poorCount"]
        res.append(commentCount)
        res.append(goodRateShow)
        res.append(generalRateShow)
        res.append(poorRateShow)
        res.append(goodCount)
        res.append(generalCount)
        res.append(poorCount)
    except:
        print "Some error occurred"
    try:

        generalComment = result["hotCommentTagStatistics"]
        for i in range(len(generalComment)):
            generalCommentDic[generalComment[i]["name"]] = generalComment[i]["count"]
    except:
        print "Some error occurred"

    return res, generalCommentDic


def getComment(code):
    import json
    comments = []
    try:
        for j in range(10):
            url = "http://club.jd.com/productpage/p-" + code + "-s-0-t-3-p-" + str(j) + ".html"
            request = urllib2.Request(url)
            response = urllib2.urlopen(request)
            content = response.read()
            content = content.decode('gbk')
            result = json.loads(content)
            comment = result["comments"]
            for i in range(len(comment)):
                comments.append(comment[i]["content"])
    except:
        print "No comments for this product!"

    return comments


'''获得商品品牌，名称，图片，价格，评价'''
'''不同商品基本信息不同，包含内容种类也不一样，用字典处理'''
def get_all_links(content, code):
    soup = bs4.BeautifulSoup(content, 'html.parser')
    #infoDict = {}
    #infoListTitle = []
    #infoListContent = []
    #info = soup.find('div', {'class': re.compile("Ptable-item")})

    '''try:
        for line in info:
            data1 = re.findall(r'<dt>(.*?)</dt>', str(line), re.S)
            data2 = re.findall(r'<dd>(.*?)</dd>', str(line), re.S)
            if (data1 != [] and data2 != []):
                for u1 in data1:
                    infoListTitle.append(u1)
                for u2 in data2:
                    infoListContent.append(u2)

        for i in range(len(infoListTitle)):
            infoDict[infoListTitle[i]] = infoListContent[i]
    except:
        print "NO TAG"
        infoDict['No Tag'] = '<~(*_*)~>'
    '''
    img = soup.find('img', attrs={'id': "spec-img"})
    price = getPrice(code)
    try:
        title = img['alt']
        imgUrl = "https:" + img['data-origin']
    except:
        title = ''
        imgUrl = ''

    percentageList, commentTagDic = getRate(code)
    comments = getComment(code)

    return title, imgUrl, price, percentageList, commentTagDic, comments


def get_all_items(content):
    try:
        soup = bs4.BeautifulSoup(content, 'html.parser')

        item0 = soup.find('div', attrs={'id': "J_goodsList"})

        itemId = item0.findAll('li', {'class': re.compile("gl-item")})
        itemList = []

        for items in itemId:
            itemNum = items['data-sku']
            itemList.append('https://item.jd.com/' + str(itemNum) + '.html')
        return itemList
    except:
        print "Error while finding item"


def add_page_to_folder(page, imgurl, title, price, percentageList, commentTagDic, comments):
    # 将网页存到文件夹里

    folder = 'LipStickPages'  # 存放网页的文件夹
    filename = valid_filename(imgurl)  # 将网址变成合法的文件名

    if not os.path.exists(folder):  # 如果文件夹不存在则新建
        os.mkdir(folder)
    f = open("LipStickPages/" + filename + ".txt", 'w')

    try:
        f.write(imgurl + '\n')
        f.write(page + '\n')
        f.write(title + '\n')
        f.write(price+'\n')
    except:
        print "Error 1"
    '''
    try:
        for i in range(len(infoDict)):
            f.write(infoDict.keys()[i] + '\n')
            f.write(str(infoDict.values()[i]) + '\n')
    except:
        print "Error 2"
    '''
    try:
        for percent in percentageList:
            #7个数字，１个总数，３个百分比，３个整数
            f.write(str(percent))
            f.write('\n')
    except:
        print "Percent Error"

    try:
        for j in range(len(commentTagDic)):
            f.write(commentTagDic.keys()[j])
            f.write('\n')
            f.write(str(commentTagDic.values()[j]))
            f.write('\n')
        f.write('\n')
        for c in comments:
            f.write(c + '\n')

    except:
        print "Comment ERROR"
    f.close()

def bloomfilter(str):
    global table
    for i in range(10):
        num = globals()['h%s' % i](str)
        num = num % (20*n)
        if table[num] == 0:
            return False

    return True


def working():
    global COUNT
    COUNT = 1
    while COUNT < n:
        page = q.get()
        '''必须从商品网址开始爬'''
        if ('item.jd.com' in page or 'keyword=%E5%8F%A3%E7%BA%A2' in page) and not bloomfilter(page):
            if ('keyword=%E5%8F%A3%E7%BA%A2' in page):
                content = get_Frontpage(page)
                outlinks = get_all_items(content)
                for link in outlinks:
                    q.put(link)
            if ('item.jd.com' in page):
                content, code = get_page(page)

                title, imgUrl, price, percentageList, commentTagDic, comments = get_all_links(content, code)
            #dd = re.compile(r'[\u4e00-\u9fa5]+', re.S)
            #ds = re.compile(r'[\s{}()a-zA-Z.,/\-%#+$&*~|!\"，\']+', re.S)


            #title = dd.sub('', str(title))

                seg_list = jieba.cut(title)
                title = ' '.join(seg_list)

                add_page_to_folder(page, imgUrl, title, price, percentageList, commentTagDic, comments)

                put_into_table(page)
                if varLock.acquire():
                    varLock.release()
                q.task_done()
                COUNT += 1
        if (q.empty()):
            break
    os._exit(0)


def put_into_table(ch):
    global table
    for i in range(10):
        num = globals()['h%s' % i](ch) % (20 * n)
        table[num] = 1


if __name__ == '__main__':
    #NUM 为增加的线程个数
    NUM = 20

    varLock = threading.Lock()
    ioLock = threading.Lock()
    q = Queue.Queue()
    #n为期望爬取的网页数量
    n = 300000
    for i in range(108):
        unitUrl = 'https://search.jd.com/Search?keyword=%E5%8F%A3%E7%BA%A2&enc=utf-8&qrst=1&rt=1&stop=1' \
                  '&vt=2&wq=%E5%8F%A3%E7%BA%A2&stock=1&page=' + str(i) + '&s=' + str(55*i) + '&click=0'
        q.put(unitUrl)

    table = [0] * 20 * n
    for i in range(NUM):
        t = threading.Thread(target=working)
        t.setDaemon(True)
        t.start()

    q.join()
