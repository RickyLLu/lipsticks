#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene
import jieba
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

def parseCommand(command):
    allowed_opt = ['title', 'brand', 'language','site','name']
    command_dict = {}
    opt = 'contents'
    for i in command.split(' '):
        if ':' in i:
            opt, value = i.split(':')[:2]
            opt = opt.lower()
            if opt in allowed_opt and value != '':
                command_dict[opt] = command_dict.get(opt, '') + ' ' + value
        else:
            lis=jieba.cut(i)
	    for j in lis:
	        command_dict[opt] = command_dict.get(opt, '') + ' ' + j
    return command_dict


def run(searcher_good, searcher_bad, analyzer):
    while True:
        command_dict = parseCommand(command)
	total_num=20

        #这些不同的s用来决定排序顺序：依次是按价格（从低到高）、热度（总评论数）、好评率、综合评分
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

        #这两句用来限定价格的范围
	#q=NumericRangeQuery.newFloatRange("price",100.0,200.0,True,True)
	#querys.add(q,BooleanClause.Occur.MUST)

        scoreDocs_good = searcher_good.search(querys, total_num,so).scoreDocs
	total=len(scoreDocs_good)
	flag=True
	if len(scoreDocs_good)<total_num:
	    scoreDocs_bad = searcher_bad.search(querys, total_num,so).scoreDocs
	    total=total+len(scoreDocs_bad)
	    flag=False
	if total>total_num:
	    total=total_num
        print "%s total matching documents." % total

        #"url"是网址，“img_url”是图片网址，“brand”是品牌
        for scoreDoc_good in scoreDocs_good:
            doc = searcher_good.doc(scoreDoc_good.doc)
##            explanation = searcher.explain(query, scoreDoc.doc)
            print "------------------------"
            print 'title:', doc.get('title')
            print 'total_comment',doc.get("total_comment")
	    print 'price',doc.get("price")
	    print 'socre',doc.get("socre")
	    print 'brand',doc.get("brand")
	    print 'good_rate',doc.get("good_rate")
	    print 
	if not flag:
	    t=0
	    for scoreDoc_bad in scoreDocs_bad:
		t=t+1
                doc = searcher_bad.doc(scoreDoc_bad.doc)
##                explanation = searcher.explain(query, scoreDoc.doc)
                print "------------------------"
                print 'title:', doc.get('title')
                print 'total_comment',doc.get("total_comment")
	        print 'price',doc.get("price")
	        print 'score',doc.get("score")
	        print 'brand',doc.get("brand")
	        print 'good_rate',doc.get("good_rate")
	        print 
		if t>total_num-1-len(scoreDocs_good):
		    break
	
##            print explanation


if __name__ == '__main__':
    STORE_DIR_good = "index_good"
    STORE_DIR_bad = "index_bad"
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print 'lucene', lucene.VERSION
    #base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    directory_good = SimpleFSDirectory(File(STORE_DIR_good))
    directory_bad = SimpleFSDirectory(File(STORE_DIR_bad))
    searcher_good = IndexSearcher(DirectoryReader.open(directory_good))
    searcher_bad = IndexSearcher(DirectoryReader.open(directory_bad))
    analyzer = WhitespaceAnalyzer(Version.LUCENE_CURRENT)
    run(searcher_good, searcher_bad, analyzer)
    del searcher








