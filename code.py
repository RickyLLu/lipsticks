#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

import web
from web import form
import urllib2
import sys, os, lucene
import jieba
import cv2
from numpy import *
import os
import dlib
import math
import face_recognition
import numpy as np
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version
from org.apache.lucene.search import BooleanQuery
from org.apache.lucene.search import BooleanClause
from org.apache.lucene.analysis.core import SimpleAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.search import SortField
from org.apache.lucene.search import Sort
from org.apache.lucene.search import NumericRangeQuery
from globals import vm_env

def parseCommand(command,brand):
    allowed_opt = ['title', 'brand', 'language','site','name']
    command_dict = {}
    opt = 'contents'
    q=command.split()
    try:
	high=float(str(q[-1]))
	q.pop()
    except:
	high=99999
    try:
	low=float(str(q[-1]))
	q.pop()
    except:
	low=0
    if brand!="1":
        command_dict['brand']=command_dict.get('brand', '') + ' ' + brand
    for i in q:
        if ':' in i:
	   
            opt, value = i.split(':')[:2]
            opt = opt.lower()
            if opt in allowed_opt and value != '':
                command_dict[opt] = command_dict.get(opt, '') + ' ' + value
        else:
            lis=jieba.cut(i)
	    for j in lis:
	        command_dict[opt] = command_dict.get(opt, '') + ' ' + j
    return command_dict,low,high


def Run_Price(searcher_good, searcher_bad, analyzer, command, brand):
    while True:
        command_dict,low,high = parseCommand(command, brand)
	total_num=20

	s=SortField("price",SortField.Type.FLOAT,False)
	#s=SortField("total_comment",SortField.Type.FLOAT,True)
	#s=SortField("good_rate",SortField.Type.FLOAT,True)
	#s=SortField("socre",SortField.Type.FLOAT,True)
	so=Sort(s)
        querys = BooleanQuery()
        for k,v in command_dict.iteritems():
            query = QueryParser(Version.LUCENE_CURRENT, k,
                              analyzer).parse(v)
            querys.add(query, BooleanClause.Occur.MUST)

        #The price's range
	q=NumericRangeQuery.newFloatRange("price",low,high,True,True)
	querys.add(q,BooleanClause.Occur.MUST)
	
        scoreDocs_good = searcher_good.search(querys, total_num,so).scoreDocs
	total=len(scoreDocs_good)
	flag=True
	if len(scoreDocs_good)<total_num:
	    scoreDocs_bad = searcher_bad.search(querys, total_num,so).scoreDocs
	    total=total+len(scoreDocs_bad)
	    flag=False
	if total>total_num:
	    total=total_num
        #Total is the number of matched websites
	res = []
        for scoreDoc_good in scoreDocs_good:
	    unit = []
            doc = searcher_good.doc(scoreDoc_good.doc)
            title = doc.get('title')
	    title.replace(' ', '')
	    title = title[:18]
            total_comment = doc.get("total_comment")
	    price = doc.get("price")
	    socre = doc.get("socre")
	    brand = doc.get("brand")
	    good_rate = doc.get("good_rate")
	    url = doc.get("url")
	    img_url = doc.get("img_url")
	    comment = doc.get("comment").split()
	    unit.append(title)             #0
	    unit.append(total_comment)     #1
	    unit.append(price)             #2
	    unit.append(socre)		   #3
	    unit.append(brand)		   #4
	    unit.append(good_rate)	   #5
	    unit.append(url)		   #6
	    unit.append(img_url)	   #7
	    unit.append(comment) 	   #8
	    res.append(unit)		  
	if not flag:
	    t=0
	    for scoreDoc_bad in scoreDocs_bad:
		t=t+1
                doc = searcher_bad.doc(scoreDoc_bad.doc)
##                explanation = searcher.explain(query, scoreDoc.doc)
                title = doc.get('title')
		title.replace(' ', '')
           	title = title[:18]
                total_comment = doc.get("total_comment")
	        price = doc.get("price")
	        socre = doc.get("socre")
	        brand = doc.get("brand")
	        good_rate = doc.get("good_rate")
		url = doc.get("url")
                img_url = doc.get("img_url")
		comment = doc.get("comment").split()
	        unit.append(title)
	    	unit.append(total_comment)
	    	unit.append(price)
	    	unit.append(socre)
	    	unit.append(brand)
	    	unit.append(good_rate)
		unit.append(url)
		unit.append(img_url)
		unit.append(comment)
	    	res.append(unit)
		if t>total_num-1-len(scoreDocs_good):
		    break
	res.append(brand)
	return res


def Run_TotalComment(searcher_good, searcher_bad, analyzer, command, brand):
    while True:
        command_dict,low,high = parseCommand(command, brand)
	total_num=20

	#s=SortField("price",SortField.Type.FLOAT,False)
	s=SortField("total_comment",SortField.Type.FLOAT,True)
	#s=SortField("good_rate",SortField.Type.FLOAT,True)
	#s=SortField("socre",SortField.Type.FLOAT,True)
	so=Sort(s)

        querys = BooleanQuery()
        for k,v in command_dict.iteritems():
            query = QueryParser(Version.LUCENE_CURRENT, k,
                              analyzer).parse(v)
            querys.add(query, BooleanClause.Occur.MUST)

        #The price's range
	q=NumericRangeQuery.newFloatRange("price",low,high,True,True)
	querys.add(q,BooleanClause.Occur.MUST)
	
        scoreDocs_good = searcher_good.search(querys, total_num,so).scoreDocs
	total=len(scoreDocs_good)
	flag=True
	if len(scoreDocs_good)<total_num:
	    scoreDocs_bad = searcher_bad.search(querys, total_num,so).scoreDocs
	    total=total+len(scoreDocs_bad)
	    flag=False
	if total>total_num:
	    total=total_num
        #Total is the number of matched websites
	res = []
        for scoreDoc_good in scoreDocs_good:
	    unit = []
            doc = searcher_good.doc(scoreDoc_good.doc)
            title = doc.get('title')
	    title.replace(' ', '')
	    title = title[:18]
            total_comment = doc.get("total_comment")
	    price = doc.get("price")
	    socre = doc.get("socre")
	    brand = doc.get("brand")
	    good_rate = doc.get("good_rate")
	    url = doc.get("url")
	    img_url = doc.get("img_url")
	    comment = doc.get("comment").split()
	    unit.append(title)             #0
	    unit.append(total_comment)     #1
	    unit.append(price)             #2
	    unit.append(socre)		   #3
	    unit.append(brand)		   #4
	    unit.append(good_rate)	   #5
	    unit.append(url)		   #6
	    unit.append(img_url)	   #7
	    unit.append(comment)	   #8
	    res.append(unit)		  
	if not flag:
	    t=0
	    for scoreDoc_bad in scoreDocs_bad:
		t=t+1
                doc = searcher_bad.doc(scoreDoc_bad.doc)
##                explanation = searcher.explain(query, scoreDoc.doc)
                title = doc.get('title')
		title.replace(' ', '')
           	title = title[:18]
                total_comment = doc.get("total_comment")
	        price = doc.get("price")
	        socre = doc.get("socre")
	        brand = doc.get("brand")
	        good_rate = doc.get("good_rate")
		url = doc.get("url")
                img_url = doc.get("img_url")
		comment = doc.get("comment").split()
	        unit.append(title)
	    	unit.append(total_comment)
	    	unit.append(price)
	    	unit.append(socre)
	    	unit.append(brand)
	    	unit.append(good_rate)
		unit.append(url)
		unit.append(img_url)
		unit.append(comment)
	    	res.append(unit)
		if t>total_num-1-len(scoreDocs_good):
		    break
	res.append(brand)
	return res


def Run_GoodRate(searcher_good, searcher_bad, analyzer, command, brand):
    while True:
        command_dict,low,high = parseCommand(command, brand)
	total_num=20

	#s=SortField("price",SortField.Type.FLOAT,False)
	#s=SortField("total_comment",SortField.Type.FLOAT,True)
	s=SortField("good_rate",SortField.Type.FLOAT,True)
	#s=SortField("socre",SortField.Type.FLOAT,True)
	so=Sort(s)

        querys = BooleanQuery()
        for k,v in command_dict.iteritems():
            query = QueryParser(Version.LUCENE_CURRENT, k,
                              analyzer).parse(v)
            querys.add(query, BooleanClause.Occur.MUST)

        #The price's range
	q=NumericRangeQuery.newFloatRange("price",low,high,True,True)
	querys.add(q,BooleanClause.Occur.MUST)
	
        scoreDocs_good = searcher_good.search(querys, total_num,so).scoreDocs
	total=len(scoreDocs_good)
	flag=True
	if len(scoreDocs_good)<total_num:
	    scoreDocs_bad = searcher_bad.search(querys, total_num,so).scoreDocs
	    total=total+len(scoreDocs_bad)
	    flag=False
	if total>total_num:
	    total=total_num
        #Total is the number of matched websites
	res = []
        for scoreDoc_good in scoreDocs_good:
	    unit = []
            doc = searcher_good.doc(scoreDoc_good.doc)
##            explanation = searcher.explain(query, scoreDoc.doc) ???
            title = doc.get('title')
	    title.replace(' ', '')
	    title = title[:18]
            total_comment = doc.get("total_comment")
	    price = doc.get("price")
	    socre = doc.get("socre")
	    brand = doc.get("brand")
	    good_rate = doc.get("good_rate")
	    url = doc.get("url")
	    img_url = doc.get("img_url")
	    comment = doc.get("comment").split()
	    unit.append(title)             #0
	    unit.append(total_comment)     #1
	    unit.append(price)             #2
	    unit.append(socre)		   #3
	    unit.append(brand)		   #4
	    unit.append(good_rate)	   #5
	    unit.append(url)		   #6
	    unit.append(img_url)	   #7
	    unit.append(comment)	   #8
	    res.append(unit)		  
	if not flag:
	    t=0
	    for scoreDoc_bad in scoreDocs_bad:
		t=t+1
                doc = searcher_bad.doc(scoreDoc_bad.doc)
##                explanation = searcher.explain(query, scoreDoc.doc)
                title = doc.get('title')
		title.replace(' ', '')
           	title = title[:18]
                total_comment = doc.get("total_comment")
	        price = doc.get("price")
	        socre = doc.get("socre")
	        brand = doc.get("brand")
	        good_rate = doc.get("good_rate")
		url = doc.get("url")
                img_url = doc.get("img_url")
		comment = doc.get("comment").split()
	        unit.append(title)
	    	unit.append(total_comment)
	    	unit.append(price)
	    	unit.append(socre)
	    	unit.append(brand)
	    	unit.append(good_rate)
		unit.append(comment)
		unit.append(url)
		unit.append(img_url)
	    	res.append(unit)
		if t>total_num-1-len(scoreDocs_good):
	       	    break
	res.append(brand)
	return res


def Run_Score(searcher_good, searcher_bad, analyzer, command, brand):
    while True:
        command_dict,low,high = parseCommand(command,brand)
	total_num=20

	#s=SortField("price",SortField.Type.FLOAT,False)
	#s=SortField("total_comment",SortField.Type.FLOAT,True)
	#s=SortField("good_rate",SortField.Type.FLOAT,True)
	s=SortField("socre",SortField.Type.FLOAT,True)
	so=Sort(s)

        querys = BooleanQuery()
        for k,v in command_dict.iteritems():
            query = QueryParser(Version.LUCENE_CURRENT, k, analyzer).parse(v)
            querys.add(query, BooleanClause.Occur.MUST)

        #The price's range
	q=NumericRangeQuery.newFloatRange("price",low,high,True,True)
	querys.add(q,BooleanClause.Occur.MUST)
	
        scoreDocs_good = searcher_good.search(querys, total_num,so).scoreDocs
	total=len(scoreDocs_good)
	flag=True
	if len(scoreDocs_good)<total_num:
	    scoreDocs_bad = searcher_bad.search(querys, total_num,so).scoreDocs
	    total=total+len(scoreDocs_bad)
	    flag=False
	if total>total_num:
	    total=total_num
        #Total is the number of matched websites
	res = []
        for scoreDoc_good in scoreDocs_good:
	    unit = []
            doc = searcher_good.doc(scoreDoc_good.doc)
##            explanation = searcher.explain(query, scoreDoc.doc) ???
            title = doc.get('title')
	    title.replace(' ', '')
	    title = title[:18]
            total_comment = doc.get("total_comment")
	    price = doc.get("price")
	    socre = doc.get("socre")
	    brand = doc.get("brand")
	    good_rate = doc.get("good_rate")
	    url = doc.get("url")
	    img_url = doc.get("img_url")
	    comment = doc.get("comment").split()
	    unit.append(title)             #0
	    unit.append(total_comment)     #1
	    unit.append(price)             #2
	    unit.append(socre)		   #3
	    unit.append(brand)		   #4
	    unit.append(good_rate)	   #5
	    unit.append(url)		   #6
	    unit.append(img_url)	   #7
	    unit.append(comment)
	    res.append(unit)		  
	if not flag:
	    t=0
	    for scoreDoc_bad in scoreDocs_bad:
		t=t+1
                doc = searcher_bad.doc(scoreDoc_bad.doc)
##                explanation = searcher.explain(query, scoreDoc.doc)
                title = doc.get('title')
		title.replace(' ', '')
           	title = title[:18]
                total_comment = doc.get("total_comment")
	        price = doc.get("price")
	        socre = doc.get("socre")
	        brand = doc.get("brand")
	        good_rate = doc.get("good_rate")
		url = doc.get("url")
                img_url = doc.get("img_url")
		comment = doc.get("comment").split()
	        unit.append(title)
	    	unit.append(total_comment)
	    	unit.append(price)
	    	unit.append(socre)
	    	unit.append(brand)
	    	unit.append(good_rate)
		unit.append(url)
		unit.append(img_url)
		unit.append(comment)
	    	res.append(unit)
		if t>total_num-1-len(scoreDocs_good):
		    break
	res.append(brand)
	return res

#Lip-color matching
def initial(folder):
    for brand in folder:
        f = open(brand + '/' + brand + '.txt', 'r')
        f2 = open(brand + '/' + 'color.txt', 'w')
        for k in range(len(os.listdir(brand + '/')) - 2):
            if brand=='ysl':
                img = cv2.imread(brand + '/' + str(k + 1) + '.jpg', cv2.IMREAD_COLOR)
            else:
                img = cv2.imread(brand + '/' + str(k + 1) + '.png', cv2.IMREAD_COLOR)
            img = cv2.GaussianBlur(img, (5,5), 0.1)
            rbins = [0]*256
            gbins = [0]*256
            bbins = [0]*256
            for i in range(len(img)):
                for j in range(len(img[0])):
                    tmp = img[i][j]
                    tmpb, tmpg, tmpr = int(round(tmp[0])), int(round(tmp[1])), int(round(tmp[2]))
                    if tmpb==0 and tmpg==0 and tmpr==0:
                        continue
                    rbins[tmpr] += 1
                    gbins[tmpg] += 1
                    bbins[tmpb] += 1
            r = rbins.index(max(rbins))
            g = gbins.index(max(gbins))
            b = bbins.index(max(bbins))
            f2.write(str(r) + '\t' + str(g) + '\t' + str(b) + '\n')
        f.close()
        f2.close()

def stdcolor(folder):
    std = {}
    for brand in folder:
        f = open(brand + '/' + brand + '.txt', 'r')
        f2 = open(brand + '/' + 'color.txt', 'r')
        for line in f2:
            tmp = line.split()
            tmp2 = f.readline().split('\t')
	    for  i in range(len(tmp2)):
		if brand=='dior' or brand=='ysl':
		    tmp2[i]=tmp2[i].decode('gbk')
		else:
		    tmp2[i]=tmp2[i].decode('utf8')
            std[tmp2[0]] = [(int(tmp[0]), int(tmp[1]), int(tmp[2])), tmp2[1:]]
        f.close()
        f2.close()
    return std

def lipscolor(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
    faces = detector(img, 1)
    feature = []
    for face in faces:
        feature.append(predictor(img, face))
    xs = []
    ys = []
    for i in range(48,68):
        xs.append(feature[0].part(i).x)
        ys.append(feature[0].part(i).y)
    r1 = 0
    g1 = 0
    b1 = 0
    r2 = 0
    g2 = 0
    b2 = 0
    count1 = 0
    count2 = 0
    for i in [1,2,4,5]:
        for j in [13,14,15]:
            d = sqrt((ys[i] - ys[j]) ** 2 + (xs[i] - xs[j]) ** 2)
            theta = 90 if xs[i] == xs[j] else math.atan((ys[i] - ys[i]) / float(xs[i] - xs[j]))
            th = max(abs(d*math.cos(theta)), abs(d * math.sin(theta)))
            for k in range(1,int(th) + 1):
                b1 += img[int(round(ys[i] + k * math.sin(theta)))][int(round(xs[i] + k * math.cos(theta)))][0]
                g1 += img[int(round(ys[i] + k * math.sin(theta)))][int(round(xs[i] + k * math.cos(theta)))][1]
                r1 += img[int(round(ys[i] + k * math.sin(theta)))][int(round(xs[i] + k * math.cos(theta)))][2]
                count1 += 1
    for i in [7,8,10,11]:
        for j in [17,19]:
            d = sqrt((ys[i] - ys[j]) ** 2 + (xs[i] - xs[j]) ** 2)
            theta = 90 if xs[i] == xs[j] else math.atan((ys[i] - ys[i]) / float(xs[i] - xs[j]))
            th = max(abs(d*math.cos(theta)), abs(d * math.sin(theta)))
            for k in range(1,int(th) + 1):
                b2 += img[int(round(ys[i] + k * math.sin(theta)))][int(round(xs[i] + k * math.cos(theta)))][0]
                g2 += img[int(round(ys[i] + k * math.sin(theta)))][int(round(xs[i] + k * math.cos(theta)))][1]
                r2 += img[int(round(ys[i] + k * math.sin(theta)))][int(round(xs[i] + k * math.cos(theta)))][2]
                count2 += 1
    r = ((r1 / count1) * 0.6 + (r2 / count2) * 0.4) * 1.01         
    g = ((g1 / count1) * 0.6 + (g2 / count2) * 0.4) * 0.68
    b = ((b1 / count1) * 0.6 + (b2 / count2) * 0.4) * 0.7
    return int(r),int(g),int(b)

def matchcolor(r, g, b, std):
    res = []
    u = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    v = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
    minimum = sqrt(255**2 *3)
    for i in std:
        #tmp = sqrt((r - std[i][0][0])**2+(g - std[i][0][1])**2+(b - std[i][0][2])**2)
        u1 = -0.1687 * std[i][0][0] - 0.3313 * std[i][0][1] + 0.5 * std[i][0][2] + 128
        v1 = 0.5 * std[i][0][0] - 0.4187 * std[i][0][1] - 0.0813 * std[i][0][2] + 128
        tmp = sqrt((u - u1) ** 2 + (v - v1) ** 2) 
        if (tmp == minimum) or (tmp > minimum and tmp < minimum + 10):
            res.append((i + '\t' + '\t'.join(std[i][1]), std[i][0][0], std[i][0][1], std[i][0][2]))
        elif tmp < minimum - 10:
            res = [(i + '\t' + '\t'.join(std[i][1]), std[i][0][0], std[i][0][1], std[i][0][2])]
            minimum = tmp
        elif tmp < minimum:
            res.insert(0, (i + '\t' + '\t'.join(std[i][1]), std[i][0][0], std[i][0][1], std[i][0][2]))
            minimum =tmp  
    return res     

def rounde(p):
    if p>255:
        p=255
    if p<0:
        p=0
    return p

def color(b,g,r,img, n):
    img2=np.zeros((len(img),len(img[0]),3),dtype=np.uint8)
    for i in range(len(img)):
        for j in range(len(img[0])):
            img2[i][j][0]=int(img[i][j][0])
            img2[i][j][1]=int(img[i][j][1])
            img2[i][j][2]=int(img[i][j][2])
    face_list=face_recognition.face_landmarks(img)
    p=[]
    for k in face_list[0]["top_lip"]:
        p.append([k[0],k[1]])
    pts=np.array(p,np.int32)
    pts=pts.reshape((-1,1,2))
    cv2.fillPoly(img2,[pts],(255,255,0))
    p=[]
    l=[]
    for k in face_list[0]["bottom_lip"]:
        l.append([k[0],k[1]])
    pts=np.array(l,np.int32)
    pts=pts.reshape((-1,1,2))
    cv2.fillPoly(img2,[pts],(255,255,0))
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img2[i][j][0]==255 and img2[i][j][1]==255 and img2[i][j][2]==0:
                p.append((i,j))
    u1=-0.147*r-0.289*g+0.436*b
    v1=0.615*r-0.515*g-0.1*b
    for k in p:
        i=k[0]
        j=k[1]
        y=0.299*int(img[i][j][2])+0.114*int(img[i][j][0])+0.587*int(img[i][j][1])
        y2=y
        u2=u1
        v2=v1
        r_new=y2+1.14*v2
        g_new=y2-0.39*u2-0.58*v2
        b_new=y2+2.03*u2
        r_new=rounde(r_new)
        g_new=rounde(g_new)
        b_new=rounde(b_new)
        img2[i][j][0]=b_new
        img2[i][j][1]=g_new
        img2[i][j][2]=r_new
    cv2.imwrite("static/img/"+str(n)+".png",img2)

#Lipstick Matching
def preprocessing():
    orb = cv2.ORB()
    for brand in folder:
        f = open(brand + '/' + brand + '.txt', 'r')
        t = len(os.listdir(brand + '/')) - 1
        for k in range(t):
            f2 = open('dataset/' + brand + str(k + 1) + '.txt', 'w')
            img = cv2.imread(brand + '/' + str(k + 1) + '.jpg', cv2.IMREAD_COLOR)
            img = cv2.GaussianBlur(img, (5,5), 0.1)
            tmp = f.readline()
            f2.write(tmp)
            des = orb.detectAndCompute(img, None)[1]
            for i in des:
                for j in i:
                    f2.write(str(j) + '\t')
                f2.write('\n')
            f2.close()
        f.close()


def initialize():
    lipstick = {}
    for i in os.listdir('dataset/'):
        f = open('dataset/' + i, 'r')
        key = f.readline()
        value = []
        for j in f.readlines():
            tmp = j.split()
            temp = []
            for k in tmp:
                temp.append(int(k))
            value.append(array(temp,dtype='uint8'))
        lipstick[key] = array(value)
    return lipstick

dataset = initialize()

def match(img, dataset):
    orb = cv2.ORB()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    img = cv2.GaussianBlur(img, (5,5), 0.1)
    tdes = orb.detectAndCompute(img, None)[1]
    res = []
    minimum = 100000000
    for i in dataset:
        tmp = 0
        matches = bf.match(tdes, dataset[i])
        for j in range(len(matches)):
            tmp += matches[j].distance
        if (tmp == minimum) or (tmp > minimum and tmp < minimum + 10):
            res.append(i)
        elif tmp < minimum -10:
            res = [i]
            minimum = tmp
        elif tmp < minimum:
            res.insert(0, i)
            minimum =tmp 
    return res

urls = (
    '/', 'HomePage',
    '/p', 'FindPrice',
    '/h', 'HotOnSale',
    '/g', 'GoodReputation',
    '/t(.*)', 'SortByScore',
    '/pt(.*)', 'SortByPrice',
    '/ht(.*)', 'SortByHot',
    '/gt(.*)', 'SortByGood',
    '/co(.*)', 'GetComments',
    '/x', 'UpLoad',
    '/v', 'ColorTest',
    '/a', 'UploadLipstick',
    '/b', 'LipstickMatch',
    '/d(.*)', 'TryMyself'
)


render = web.template.render('templates/', cache = False) # your templates      

    
def yourInput(command):	
    return command


class HomePage:
    def GET(self):
        return render.HomePage()

class FindPrice:
    def GET(self):
        return render.FindPrice()

class HotOnSale:
    def GET(self):
	return render.HotOnSale()

class GoodReputation:
    def GET(self):
	return render.GoodReputation()

class SortByScore:
    def GET(self, name):
	STORE_DIR_GOOD = "index_good"
	STORE_DIR_BAD = "index_bad"
        vm_env.attachCurrentThread()
        directory_good = SimpleFSDirectory(File(STORE_DIR_GOOD))
        searcher_good = IndexSearcher(DirectoryReader.open(directory_good))
	directory_bad = SimpleFSDirectory(File(STORE_DIR_BAD))
        searcher_bad = IndexSearcher(DirectoryReader.open(directory_bad))
        analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
        user_data = web.input(name = None)
        command = yourInput(user_data.shop)
        #command=command+u' '+u'brand:'+xx.decode('utf8')
	res = Run_Score(searcher_good, searcher_bad, analyzer, command, user_data.brand)
	res.append(command)
        return render.SearchResult(res)

class SortByPrice:
    def GET(self, name):
	STORE_DIR_GOOD = "index_good"
	STORE_DIR_BAD = "index_bad"
        vm_env.attachCurrentThread()
        directory_good = SimpleFSDirectory(File(STORE_DIR_GOOD))
        searcher_good = IndexSearcher(DirectoryReader.open(directory_good))
	directory_bad = SimpleFSDirectory(File(STORE_DIR_BAD))
        searcher_bad = IndexSearcher(DirectoryReader.open(directory_bad))
        analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
        user_data = web.input(name = None)
        command = yourInput(user_data.shop)
	res = Run_Price(searcher_good, searcher_bad, analyzer, command, user_data.brand)
	res.append(command)
        return render.SearchResult(res)

class SortByHot:
    def GET(self, name):
	STORE_DIR_GOOD = "index_good"
	STORE_DIR_BAD = "index_bad"
        vm_env.attachCurrentThread()
        directory_good = SimpleFSDirectory(File(STORE_DIR_GOOD))
        searcher_good = IndexSearcher(DirectoryReader.open(directory_good))
	directory_bad = SimpleFSDirectory(File(STORE_DIR_BAD))
        searcher_bad = IndexSearcher(DirectoryReader.open(directory_bad))
        analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
        user_data = web.input(name = None)
        command = yourInput(user_data.shop)
	res = Run_TotalComment(searcher_good, searcher_bad, analyzer, command, user_data.brand)
	res.append(command)
        return render.SearchResult(res)

class SortByGood:
    def GET(self, name):
	STORE_DIR_GOOD = "index_good"
	STORE_DIR_BAD = "index_bad"
        vm_env.attachCurrentThread()
        directory_good = SimpleFSDirectory(File(STORE_DIR_GOOD))
        searcher_good = IndexSearcher(DirectoryReader.open(directory_good))
	directory_bad = SimpleFSDirectory(File(STORE_DIR_BAD))
        searcher_bad = IndexSearcher(DirectoryReader.open(directory_bad))
        analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
        user_data = web.input(name = None)
        command = yourInput(user_data.shop)
	res = Run_GoodRate(searcher_good, searcher_bad, analyzer, command, user_data.brand)
	res.append(command)
        return render.SearchResult(res)

class GetComments:
    def GET(self, name):
	STORE_DIR_GOOD = "index_good"
	STORE_DIR_BAD = "index_bad"
        vm_env.attachCurrentThread()
        directory_good = SimpleFSDirectory(File(STORE_DIR_GOOD))
        searcher_good = IndexSearcher(DirectoryReader.open(directory_good))
	directory_bad = SimpleFSDirectory(File(STORE_DIR_BAD))
        searcher_bad = IndexSearcher(DirectoryReader.open(directory_bad))
        analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
	user_data = web.input(name = None)
        command = yourInput(user_data)
	if user_data.brand == '':
	    user_data.brand = '1'
	res = Run_Score(searcher_good, searcher_bad, analyzer, name, user_data.brand)
	comments = []
	for i in range(len(res)):
	    if len(res[i])==9:
	        t = res[i][8]
	    else:
		t = ''
	    for j in range(len(t)):
		s = t[j]
	        s.encode("utf8")
		if len(s) >= 50:
    	    	    comments.append(s)
        return render.comments(comments)

class UpLoad:
    def GET(self):
	return render.UpLoad()

class ColorTest:
    def GET(self): 
	img = cv2.imread("/home/jerrry/PycharmProjects/Engine1.0/static/img/picture.jpg", cv2.IMREAD_COLOR)
	std = stdcolor(folder)
	r,g,b = lipscolor(img)
	res = matchcolor(r,g,b,std)
	ColorList = []
	TList = []
	rList = []
	gList = []
	bList = []
	nameList= []
	cList = []
	for i in res:
	    tmp = i[0].split("\t")
	    cList.append(tmp[1])
	    s = u''
	    for j in range(len(tmp[0])):
	        try:
	            s += tmp[0][j]
	        except:
		    s += ''
	    nameList.append(s)
	    ColorList.append(tmp[-2])
	    TList.append(tmp[-1])
	    rList.append(i[-3])
	    gList.append(i[-2])
	    bList.append(i[-1])
	ColorList = ColorList[:5]
	TList = TList[:5]
	return render.ImageMatch(ColorList, TList, nameList, rList, gList, bList, cList)

class TryMyself:
    def GET(self, name):
	img = cv2.imread("/home/jerrry/PycharmProjects/Engine1.0/static/img/user.jpg", cv2.IMREAD_COLOR)
	user_data = web.input(name = None)
	r = int(user_data.r)
	g = int(user_data.g)
	b = int(user_data.b)
	color(b, g, r, img, 0)
	return render.TryMyself()


class UploadLipstick:
    def GET(self):
	return render.UploadLipstick()

class LipstickMatch:
    def GET(self):
	img = cv2.imread('/home/jerrry/PycharmProjects/Engine1.0/static/img/Lipstick.jpg', cv2.IMREAD_COLOR)
	res = match(img, dataset)
	res = res[0]
	tmp = res.split('\t')
	name=  tmp[0]
	try:
    	    name = name.decode("GBK")
	except:
	    name = name.decode("utf8")
	url = tmp[3]
	return render.LipstickMatch(name, url)


folder = ['dior', 'ysl', 'lancome', 'chanel']
detector = dlib.get_frontal_face_detector()
if __name__ == "__main__":    
    reload(sys)
    sys.setdefaultencoding('utf8')
    app = web.application(urls, globals())
    app.run()












