#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene, threading, time
from datetime import datetime
from bs4 import BeautifulSoup
import jieba

from java.io import File
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType, FloatField
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
import urllib
from org.apache.lucene.analysis.core import SimpleAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer


"""
This class is loosely based on the Lucene (java implementation) demo class 
org.apache.lucene.demo.IndexFiles.  It will take a directory as an argument
and will index all of the files in that directory and downward recursively.
It will index on the file path, the file name and the file contents.  The
resulting Lucene index will be placed in the current directory and called
'index'.
"""
brand_list=l=['mac','ysl','dior','lancome','channel','larastyle']
class Ticker(object):

    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)

class IndexFiles(object):
    """Usage: python IndexFiles <doc_directory>"""

    def __init__(self, root, storeDir_good,storeDir_bad, analyzer):

        if not os.path.exists(storeDir_good):
            os.mkdir(storeDir_good)
	if not os.path.exists(storeDir_bad):
            os.mkdir(storeDir_bad)

        store_good = SimpleFSDirectory(File(storeDir_good))
	store_bad = SimpleFSDirectory(File(storeDir_bad))
        analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        config = IndexWriterConfig(Version.LUCENE_CURRENT, analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
	config1 = IndexWriterConfig(Version.LUCENE_CURRENT, analyzer)
        config1.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer_good = IndexWriter(store_good, config)
	writer_bad = IndexWriter(store_bad, config1)

        self.indexDocs(root, writer_good,writer_bad)
        ticker = Ticker()
        print 'commit index',
        threading.Thread(target=ticker.run).start()
        writer_good.commit()
        writer_good.close()
	writer_bad.commit()
        writer_bad.close()
        ticker.tick = False
        print 'done'

    def indexDocs(self, root, writer_good,writer_bad):

        t1 = FieldType()
        t1.setIndexed(True)
        t1.setStored(True)
        t1.setTokenized(False)
        t1.setIndexOptions(FieldInfo.IndexOptions.DOCS_AND_FREQS)
	
	
        t2 = FieldType()
        t2.setIndexed(True)
        t2.setStored(True)
        t2.setTokenized(True)
        t2.setIndexOptions(FieldInfo.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        t3 = FieldType()
        t3.setIndexed(True)
        t3.setStored(False)
        t3.setTokenized(False)
        t3.setIndexOptions(FieldInfo.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        
	brand_list=[]
	g=open('brand.txt','r')
	for i in g.readlines():
    	    brand_list.append(i.strip())	
	socre=0
	rootdir = 'LipStickPages'
	list = os.listdir(rootdir)
	for i in range(0,len(list)):
    	    path = os.path.join(rootdir,list[i])
            if os.path.isfile(path):
                socre=0
                l=[]
                file = open(path)
                for i in file.readlines():
                    l.append((i))
                file.close()
                doc = Document()
                url=l[1]
                img_url=l[0]
                title=l[2]
                title_list=title.lower().decode('utf-8').split()
                brand=''
                for b in title_list:
                    if b in brand_list:
                        brand=b
                        break
                price=float(l[3])
                good_comment=int(l[8])
                normal_comment=int(l[9])
                bad_comment=int(l[10])
		total_comment=good_comment+normal_comment+bad_comment
		if total_comment==0:
		    continue
                good_rate=float(good_comment)/total_comment
                bad_rate=float(bad_comment)/total_comment
                socre=(float(good_comment)-5*float(bad_comment))/price
		if brand!='':
		    socre=socre*5
                doc.add(Field("title", title, t1))
                doc.add(Field("brand", brand, t1))
                doc.add(Field("url", url, t1))
                doc.add(Field("img_url", img_url, t1))
                doc.add(FloatField("total_comment",float(total_comment),Field.Store.YES))
                doc.add(FloatField("socre",float(socre),Field.Store.YES))
                doc.add(FloatField("price",float(price),Field.Store.YES))
                doc.add(FloatField("good_rate",float(good_rate),Field.Store.YES))
                doc.add(FloatField("good_comment",float(good_comment),Field.Store.YES))
		
                content=title.decode('utf-8')
                for i in range(len(l)-13):
		    content=content+' '.join(jieba.cut(l[i+12]))
                doc.add(Field("contents", content, t2))
		if price>68:
		    writer_good.addDocument(doc)
		else:
		    writer_bad.addDocument(doc)

if __name__ == '__main__':
    """
    if len(sys.argv) < 2:
        print IndexFiles.__doc__
        sys.exit(1)
    """
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print 'lucene', lucene.VERSION
    start = datetime.now()
    analyzer = WhitespaceAnalyzer(Version.LUCENE_CURRENT)
    IndexFiles('testfolder', "index_good","index_bad", analyzer)
    end = datetime.now()
    print end - start
