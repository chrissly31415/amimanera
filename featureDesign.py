#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  crawl data
"""

import pandas as pd

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
    #boiler plate length
    tmpdf=olddf.body.str.len()
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['boilerplate_length']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
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
    tmpdf=olddf.url.str.contains('www')
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['url_contains_www']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
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