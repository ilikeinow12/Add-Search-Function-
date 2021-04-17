#!/usr/bin/env python
# coding: utf-8

# ## Intro 
# ### 1) 포트폴리오 소개/ 분석 목적
# - wconcept 쇼핑몰에서 유저가 구매를 원하는 상품의 정보를 미리 받아 해당 상품을 추천해주는 기능 제안 
# 
# ### 2) 문제 제기 
# - wconcept 쇼핑몰의 화장품군 상품 노출의 비효율성 : 가장 인기가 많은 5개 내외의 상품만 지속적으로 노출됨 = 다양한 상품이 노출되고 있지 않음.
# - wconcept은 최근 화장품 분야로 사업을 대폭 확장중인 것으로 파악되는데, 기대 이상의 효과를 가지기 위해서는 더 정교한 상품 노출이 필요할 것으로 사료됨 
# - 다른 상품들과 달리 화장품의 경우 유저의 바람대로 상품을 특정하는 것이 중요함. 따라서 미리 유저의 바람을 파악하고 이에 맞추어 상품을 노출시킨다면 유저의 상품 구매 만족도는 올라가고 화장품 매출 또한 증가할 것으로 파악됨 
# 
# 
# 
# ### 3) 분석 과정 
# #### 데이터의 수집 : 웹크롤링
# - 크롤링(1) : 각 화장품의 id, 이름, 브랜드명, 가격, 리뷰수 
# - 크롤링(2) : 각 화장품의 리뷰 페이지에서 리뷰를 남긴 유저들의 피부타입과 상품의 발림성, 자극도, 향, 끈적임에 대한 평가항목 
# 
# #### 데이터의 분석 : 비지도, 지도학습 머신러닝
# - 1) 상품 Clustering : 수집한 유저의 피부타입, 화장품의 발림성, 자극도 등을 KMeans를 통해 총 5개의 유형으로 군집화 
# - 2) 추천 알고리즘에 적합한 예측모델 선정 : 위에서 얻은 table을 여러 지도학습 머신러닝에 넣어 가장 적합한 예측 모델 선정 
# - 3) 임의의 유저 정보(X)를 넣어 모델 최종 확인 : 임의의 유저를 설정하여 피부타입, 유저가 원하는 상품의 발림성, 자극도 등을 넣어 해당되는 군집의 상품을 랜덤으로 추천 
# 
# ### 4) 분석 결론 
# - 화장품의 경우 많은 유저들이 자신의 피부타입에 맞춰서 구매하고 화장품의 질감이나 자극도 등을 꼼꼼하게 따지는 경향이 강함. 따라서 미리 유저가 자신의 정보를 주게끔 유도해서 그에 맞춰 상품군을 추천한다면 기초화장품군의 판매 증가를 기대해볼 수 있음 
# 

# ## Data Import

# In[ ]:


import pandas as pd 
import pymysql
import urllib.request as req 
from bs4 import BeautifulSoup as bs
import sqlite3 
import pandas as pd 
import numpy as np 


conn = sqlite3.connect('./test.db')
cursor = conn.cursor()


# 크롤링한 데이터를 담을 테이블(1) 생성 
q = '''create table if not exists skincare (
    pdt_id char(50),
    product_name char(50),
    brand_name char(50),
    price int,
    review_num int
);
'''

cursor.execute(q)
conn.commit()


#웹크롤링 (1)
#wconcept 사이트의 기초화장품군으로 분륜된 상품의 상품 id, 상품명, 브랜드, 가격, 리뷰수 수집 

url='https://www.wconcept.co.kr/Beauty/001001'
params='?page='

limit = 25


for no in range(0, limit+1) :
    pdt_id=[]
    conn_url=f'{url}{params}{no}'
    
    html=req.urlopen(conn_url)
    soup=bs(html, 'html.parser')
    
    
    # 1)상품ID 
    lists = soup.select('div.thumbnail_list >ul >li')
    for l in lists : 
        pdt_id_find = l.a['href']
        pdt_id_int = int(pdt_id_find.split('/')[-1]) 
        pdt_id.append(pdt_id_int)
    
    
    #1)상품명  
    product_name_find = soup.findAll('div', attrs={'class':'ellipsis'})
    product_name_text=[tag.text for tag in product_name_find]

        
    
    # 2)브랜드 
    brand_name_find = soup.findAll('div', attrs={'class':'brand'})
    brand_name_text=[tag.text for tag in brand_name_find]
    

    
    #3)가격 
    price_find = soup.find_all('span', attrs={'class':'discount_price'})
    price_str = [tag.text for tag in price_find]
    
    
    #4)리뷰수 
    review_box=soup.findAll('div', attrs={'class':'review_box'}) 
    
    for i in range(0, len(product_name_text)) : 
        product_name=product_name_text[i]
        brand_name=brand_name_text[i]
        price=int(price_str[i].replace(',',''))
        
        if len(review_box[i])==1 : 
            review_num=0
           
        elif len(review_box[i])!=1:
            review_find=review_box[i].find('div', attrs={'class':'review_count'}).text[:].replace(',','')
            review_num=int(review_find)
        
        
        sql=f"""
            insert into skincare(pdt_id, product_name,brand_name, price, review_num)
            values ({pdt_id[i]}, "{product_name}", "{brand_name}", {price}, {review_num})
            """
            
        cursor.execute(sql)  
  

conn.commit()



# 크롤링한 데이터를 담을 테이블(2) 생성 
q = '''create table if not exists review_detail (
    pdt_id char(50),
    review_detailquestion char(50),
    response char(50),
    response_percent float
);
'''
cursor.execute(q)
conn.commit()

# 웹크롤링(2) 
# 각 상품마다 리뷰를 남긴 유저의 피부타입, 상품에 대한 유저의 평가항목 수집 

pdt_id=[]
#  총 26페이지의 상품 리뷰 크롤링 
for n in range(1,27) : 
    url = 'https://www.wconcept.co.kr/Beauty/001001?page={}'.format(n)
    html=req.urlopen(url)
    soup=bs(html, 'html.parser')
    
    
    # 각 페이지에 나타나는 90개의 상품 pdt_id 리스트로 저장 
    lists = soup.select('div.thumbnail_list >ul >li')
    for l in lists : 
        pdt_id_find = l.a['href']
        pdt_id_int = int(pdt_id_find.split('/')[-1]) 
        pdt_id.append(pdt_id_int)
        
        
    # 개별 상품 리뷰 수집 
   
# driver = webdriver.Chrome('./chromedriver') 
for m in range(2000, len(pdt_id)) :
    
    url = 'https://www.wconcept.co.kr/Product/{}'.format(pdt_id[m])
    html=req.urlopen(url)
    soup=bs(html, 'html.parser')
    try :       
        pdt_detail = []
        categories=soup.select('div.pdt_review--score')
        for category in categories : 
            question = category.h5.get_text()                  # 평가항목(향은 좋은가요? 등) 
            details=category.select('div.pdt_review--per')
            for detail in details :
                re=detail.strong.get_text()                    # 좋음/보통/좋지않음 
                re_percent=int(detail.em.get_text()[:-1])      # % 제외하고 숫자만  
                pdt_detail.append((pdt_id[m], question, re, re_percent))                
        
        sql = """insert into review_detail()
                 values (%s, %s, %s, %s)
              """
              
        cursor.executemany(sql, pdt_detail)
        
    except :
        pass
        

conn.commit()




#크롤링한 두 테이블을 pdt_id를 기준으로 조인 

q= ''' 
   select s.*, r.review_detailquestion, r.response, r.response_percent
   from skincare s JOIN review_detail r 
   on s.pdt_id=r.pdt_id;
   '''

df=pd.read_sql(q, conn)


# ## Data Preprocessing 

# In[1]:





# 응답항목[response] 수치화, 범주화 작업
# 구매자들은 리뷰를 남길 때 각 상품에 대해 다음의 항목을 평가하는데 이를 수치화 혹은 범주화함 
# 향/발림성은 좋은가요? (좋음, 보통, 좋지않음)  -> 2,1,0
# 제품에 끈적임이 있나요? (없음, 적당함, 심함)  -> 2,1,0
# 자극도는 어떤가요? (순함, 적당함, 자극적임)   -> 2,1,0
# 어떤 피부타입이세요? (건성, 지성, 복합성, 민감성)-> a,b,c,d
# 수치화, 범주화한 값을 [response_re]에 저장  

rate_dict = {
    '좋음':2,'보통':1,'좋지않음':0,
    '순함':2,'적당함':1,'자극적임':0,
    '심함':0,'없음':2,
     '건성':'a',
     '지성':'b',
     '복합성':'c',
     '민감성':'d'
}
df['response_re'] = df['response'].map(rate_dict)  


# 각 평가항목을 점수화하기 위한 작업 
# 평가항목에 대한 응답이 없는 경우 (response_re= null, response_percent=0에 해당되는 행) 삭제 
df['response_percent'].replace(0, np.nan, inplace=True)   #0-> null값으로 
df2=df.dropna(subset=['response_percent', 'response_re'], how='any', axis=0)

#발색력, 피부톤 항목 삭제 -> 색조화장품이 기초화장품으로 잘못 분류된 것으로 보임. 
#분석하고자 하는 데이터는 기초화장품군이므로 색조화장품은 불필요. 
df2=df2[~df2.review_detailquestion.str.contains('발색력')]
df2=df2[~df2.review_detailquestion.str.contains('피부톤')]
df2=df2[~df2.review_detailquestion.str.contains('지속력')]


# rate 컬럼 추가 : 평가항목에 대한 총점수 ([response_re] * 응답률(%)). 피부타입은 그대로 a,b,c,d값으로 전달 
# ex) 발림성은 어떤가요?에 대해 '좋음(2점)'이라는 응답률이 200%일 경우 : 2*200 = 400 
# ex) 어떤 피부타입이세요? 에 대해 '건성(a)'이라고 응답한 경우 : a 
df2['response_re_num'] = pd.to_numeric(df2.response_re, errors='coerce')
df2['rate']=df2.response_re_num*df2.response_percent
df2['rate'].fillna(df2.response_re, inplace=True)



# 위의 테이블에서 분석에 필요한 컬럼들만 따로 추출해서 새로 데이터셋 생성
# 각 화장품의 발림성, 자극도, 끈적임, 향, 리뷰를 남긴 유저들의 피부타입, 화장품의 가격 
df3 = df2.pivot_table(index=['pdt_id'],columns=['review_detailquestion'], values=['rate'], aggfunc='first')
category= ['발림성', '피부타입','자극도','끈적임','향']
df3.columns=category

df3_1 = df2.pivot_table(index=['pdt_id'], values=['price'])

df3_2=pd.merge(df3,df3_1,left_on='pdt_id', right_on='pdt_id')
df3_2.rename(columns={'price':'가격'}, inplace=True)



# 어느 하나라도 결측치가 있는 행 drop 
df3_2=df3_2.dropna(subset=['자극도', '피부타입', '끈적임', '향'], how='any', axis=0)
df3_2.info()



#피부타입 -> get dummies이용하여 encoding
onehot_skintype=pd.get_dummies(df3_2['피부타입'], prefix='skintype')
df4=pd.concat([df3_2, onehot_skintype], axis=1)
df4.drop(['피부타입'], axis=1, inplace=True)
df4.head(20)

#자료형 -> int로 
df4[['발림성', '자극도', '끈적임', '향']]=df4[['발림성', '자극도', '끈적임', '향']].astype('int')


#Kmeans를 위한 최종 데이터셋 -> X_KM 테이블에 저장 
q = '''drop table if exists X_KM;
'''
cursor.execute(q)
conn.commit()

df4.to_sql('X_KM', conn)


# ## KMeans 를 통해 상품을 5개 유형으로 군집화

# In[ ]:


# 데이터셋 X_KM 불러오기 
df_km = pd.read_sql('select * from X_KM;',conn, index_col='pdt_id')


#분석을 위한 scaling 
from sklearn import preprocessing 
X = preprocessing.MinMaxScaler().fit(df_km).transform(df_km)


#Kmeans 
from sklearn import cluster 
kmeans = cluster.KMeans(init='k-means++', n_clusters=5, n_init=10)
kmeans.fit(df_km)
cluster_label=kmeans.labels_ 
df_km['Cluster']=cluster_label


#Kmeans결과 -> final 테이블에 저장 
q = '''drop table if exists final;
'''
cursor.execute(q)
conn.commit()

df_km.to_sql('final', conn)


# ## 머신러닝 모델링

# In[115]:


df_final=pd.read_sql('select * from final;', conn, index_col='pdt_id')


X=df_final.iloc[:, :-1]
y=df_final.iloc[:, -1]

from sklearn.preprocessing import MinMaxScaler 
scaler_mm = MinMaxScaler() 
X = scaler_mm.fit_transform(X)

# Data Split 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y)


# 모델 성능평가 
# Ensemble : Voting 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier

knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
lr_model = LogisticRegression(C=1.0, class_weight='balanced', multi_class='ovr', n_jobs=-1)
dt_model = DecisionTreeClassifier(random_state=11, max_depth=3, class_weight='balanced')

from sklearn.ensemble import VotingClassifier 
ensemble_model = VotingClassifier(estimators=[('knn', knn_model),
                                             ('lr', lr_model),
                                             ('dt', dt_model)],
                                 voting='soft',
                                 n_jobs=-1)

train_scores=[m.fit(X_train, y_train).score(X_train, y_train)
             for m in [knn_model, lr_model, dt_model, ensemble_model]]
print(f'train score : \n {train_scores}')


test_scores=[m.fit(X_test, y_test).score(X_test, y_test)
             for m in [knn_model, lr_model, dt_model, ensemble_model]]
print(f'test score : \n {test_scores}')

# train score : 
# [0.9624183006535948, 0.9509803921568627, 0.9575163398692811, 0.9836601307189542]
#test score : 
# [0.9411764705882353, 0.9411764705882353, 0.9869281045751634, 0.9934640522875817]





# Random Forest 
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=5,
    random_state=1,
    n_jobs=-1).fit(X_train, y_train)

print(f'train score : {model.score(X_train, y_train)}')
print(f'test score : {model.score(X_test, y_test)}')

#train score : 0.9836601307189542
#test score : 0.9477124183006536



# XGBoost 
import xgboost as xgb 
xgb_model = xgb.XGBClassifier(
            n_estimators=10000,
            max_depth=5,
            subsample=0.5,
            reg_alpha=10,
            random_state=1).fit(X_train, y_train)


print(f'train score : {xgb_model.score(X_train, y_train)}')
print(f'test score : {xgb_model.score(X_test, y_test)}')
#train score : 0.9689542483660131
#test score : 0.9477124183006536


## 모델의 선택  :  VotingClassifier 


# ## 새로운 임의의 X 데이터 생성 후 모델 시험

# In[126]:


def recommend(n) : 
    rec_pdt_id=[]
    for n in pred : 
        rec_pdt_id = y.index[y==n]

    rec_pdt_one=np.random.choice(rec_pdt_id, 1)
    print(f'다음 상품은 어떤가요? : {rec_pdt_one}')

#유저가 각 설문 항목에 대해 자신이 원하는 정보를 입력하면 array배열로 변환하고 해당 군집의 상품을 랜덤으로 하나 추출 
# 발림성, 자극도, 끈적임, 향, 가격, 건성, 지성, 복합성, 민감성 
X_temp1=np.array([200,100,100,100, 30000, 1, 0, 0, 0]).reshape(1,-1)
pred=ensemble_model.predict(X_temp1)

recommend(pred)


# In[ ]:




