#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  crawl data
"""

import pandas as pd
from nltk.tag import pos_tag
from nltk.tag.simplify import simplify_wsj_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import random

def posTagging(olddf):
    """
    Creates new features
    """
    print "Use nltk postagging..."
    tutto=[]
    taglist=['N','NP','ADJ','ADV','PRO','V','NUM','VD','DET','P','WH','MOD','TO','VG','CNJ']
    #taglist=['N','NP','ADJ','ADV']
    
    #olddf = olddf.ix[random.sample(olddf.index, 10)]
    olddf=pd.DataFrame(olddf['body'])
    
    print type(olddf)
    for ind in olddf.index:
	  print ind
	  row=[]
	  row.append(ind)
	  text=olddf.ix[ind,'body']
	  tagged=pos_tag(word_tokenize(text))
	  tagged = [(word, simplify_wsj_tag(tag)) for word, tag in tagged]
	  tag_fd = FreqDist(tag for (word, tag) in tagged)
	  #print tagged
	  #print len(tagged)
	  
	  for l in taglist:
	      f= tag_fd[l]/float(len(tagged))
	      #print f
	      row.append(f)
	
	 #tag_fd.plot(cumulative=False)
	 # raw_input("HITKEY")
    
    
    #for index,row in pd.DataFrame(olddf['body']).iterrows():
	#tagged=pos_tag(word_tokenize(str(row)))
	#tag_fd = FreqDist(tag for (word, tag) in tagged)
	#print tag_fd.keys()
	
	#tag_fd.plot(cumulative=True)
	  tutto.append(row)
    newdf=pd.DataFrame(tutto).set_index(0)
    newdf.columns=taglist
    print newdf.head(20)
    print newdf.describe()
    newdf.to_csv("../stumbled_upon/data/postagged.csv")


def featureEngineering(olddf):
    """
    Creates new features
    """
    print "Feature engineering..."
    #lower
    olddf['url']=olddf.url.str.lower()
    olddf['embed_ratio']=olddf.embed_ratio.replace(-1.0,0.0)
    olddf['image_ratio']=olddf.image_ratio.replace(-1.0,0.0)
    olddf['is_news']=olddf['is_news'].fillna(0)
    olddf['news_front_page']=olddf['news_front_page'].fillna(0)
    #url length
    tmpdf=olddf.url.str.len()
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['url_length']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
    #body plate length
    tmpdf=olddf.body.str.len()
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['body_length']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
    #boiler plate length
    #tmpdf=olddf.boilerplate.str.len()
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['boilerplate_length']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #counts exclamation marks
    #tmpdf=olddf.body.str.count('!')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['excl_mark_number']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #counts exclamation marks
    #tmpdf=olddf.body.str.count('\\?')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['quest_mark_number']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains .com
    #tmpdf=olddf.url.str.contains('\.com')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_com']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains .org
    #tmpdf=olddf.url.str.contains('\.org')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_org']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains .co.uk
    #tmpdf=olddf.url.str.contains('\.co\.uk')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_co_uk']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
     #endswith .com
    #tmpdf=olddf.url.str.contains('com.$')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_endswith_com']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains blog
    #tmpdf=olddf.url.str.contains('blog')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_blog']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
     #contains recipe & co.
    tmpdf=olddf.url.str.contains('recipe|food|meal|kitchen|cook|apetite|meal|cuisine')
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['url_contains_foodstuff']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
     #contains recipe & co.
    tmpdf=olddf.url.str.contains('recipe')
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['url_contains_recipe']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains sweet stuff
    tmpdf=olddf.url.str.contains('cake|baking|apple|sweet|cookie|brownie|chocolat')
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['url_contains_sweetstuff']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains diet
    #tmpdf=olddf.url.str.contains('diet|calorie|nutrition|weight|fitness')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_dietfitness']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
     #contains recipe & co.
    #tmpdf=olddf.url.str.contains('recipe')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_recipe']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
     #contains health
    tmpdf=olddf.url.str.contains('health|fitness|exercise')
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['url_contains_health']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains www
    #tmpdf=olddf.url.str.contains('www')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_www']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains news
    tmpdf=olddf.url.str.contains('news|cnn')
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['url_contains_news']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains obscene
    #tmpdf=olddf.url.str.contains('obscene|sex|nude|fuck|asshole')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_obscene']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #girls
    #tmpdf=olddf.url.str.contains('girls|nude|sex|nipple')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_girls']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #syria
    #tmpdf=olddf.boilerplate.str.contains('syria|damaskus|bashar|assad')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['body_contains_syria']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    return olddf