#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  crawl data
"""

import pandas as pd
import re

def crawlRawData(lXall):
      """
      crawling raw data
      """
      print "Crawling raw data..."
      basedir='../stumbled_upon/raw_content/'
      pfacebook = re.compile("www.+facebook.+com")
      pfacebook2 = re.compile("developers.+facebook.+com.+docs.+reference.+plugins.+like|facebook.+com.+plugins.+like")
      plinkedin = re.compile("platform\.linkedin\.com")
      ptwitter = re.compile("twitter\.com\.share")
      prss=re.compile("rss feed",re.IGNORECASE)
      pgooglep=re.compile("apis\.google\.com")
      pstumble=re.compile("www\.stumbleupon\.com")
      pshare=re.compile("sharearticle|share.{1,20}article",re.IGNORECASE)
      plang=re.compile("en-US|en_US",re.IGNORECASE)
      tutto=[]
      for ind in lXall.index:
	  row=[]
	  nl=lXall.ix[ind,'numberOfLinks']
	  nl=1+lXall.ix[ind,'non_markup_alphanum_characters']
	  print "numberOfLinks:",nl
	  with open(basedir+str(ind), 'r') as content_file:
	    content = content_file.read()
	    #print "id:",ind,
	    row.append(ind)
	    
	    res = pfacebook.findall(content)
	    row.append(len(res)/float(nl))
	    
	    res = pfacebook2.findall(content)	    
	    row.append(len(res)/float(nl))
	    
	    #res = pfacebook3.findall(content)	    
	    #row.append(len(res))
	
	    #res = plinkedin.findall(content)	    
	    #row.append(len(res))
	    
	    res = ptwitter.findall(content)
	    row.append(len(res)/float(nl))
	
	    
	    res = prss.findall(content)
	    row.append(len(res)/float(nl))
	    
	    res = pgooglep.findall(content)	    
	    row.append(len(res)/float(nl))
	    
	    res = pstumble.findall(content)	    
	    row.append(len(res)/float(nl))
	    
	    m = plang.search(content)
	    if m:
		row.append(1)
	    else:
		row.append(0)
	    
	    #if len(res)>0:
		#print ind,": ",res
		#raw_input("HITKEY")

		
	    #res = pshare.findall(content)
	    #row.append(len(res)/float(nl))
	  #print ""
	  tutto.append(row)
      newdf=pd.DataFrame(tutto).set_index(0)
      newdf.columns=['wwwfacebook_ratio','facebooklike_ratio','twitter_ratio','rss_ratio','gplus_ratio','stumble_ratio','langUS']
      print newdf.head(20)
      print newdf.describe()
      return newdf