# 한국해양개발원 뉴스 토픽 모델링

## 한국해양개발원 
22.04.29~05.13

**2019-2021년도까지 3년간의 해양, 해운과 관련된 데이터 110만개의 뉴스 MongoDB에 쌓음**

- MongoDB에 쌓인 특정 키워드 검색(조선, 해양, 해운)
- 토픽 랭크를 위한 LDA 및 Word2Vec 모델 비교

# 1. LDA**란?**

<aside>
⚠️ 잠재 디리클레 할당(Latent Dirichlet Allocation, LDA)은 토픽 모델링의 대표적인 알고리즘이며, 
LDA는 각 단어나 문서의 숨겨진 주제를 찾아내어 문서와 키워드별로 주제끼리 묶어주는 
비지도 학습 알고리즘

</aside>

- LDA는 문서들은 토픽들의 혼합으로 구성되어져 있으며, 토픽들은 확률 분포에 기반하여 단어들을 생성한다고 가정 → 데이터가 주어지면, LDA는 문서가 생성되던 과정을 역추적함.
    - 빈도수 기반의 표현 방법인 BoW의 행렬 DTM 또는 TF-IDF 행렬을 입력으로 하는데, 이로부터 알 수 있는 사실은 **LDA는 단어의 순서는 신경쓰지 않겠다**

**1) 문서에 사용할 단어의 개수 N을 정합니다.**
- Ex) 5개의 단어를 정하였습니다.

**2) 문서에 사용할 토픽의 혼합을 확률 분포에 기반하여 결정합니다.**
- Ex) 위 예제와 같이 토픽이 2개라고 하였을 때 강아지 토픽을 60%, 과일 토픽을 40%와 같이 선택할 수 있습니다.

**3) 문서에 사용할 각 단어를 (아래와 같이) 정합니다.**

**3-1) 토픽 분포에서 토픽 T를 확률적으로 고릅니다.**
- Ex) 60% 확률로 강아지 토픽을 선택하고, 40% 확률로 과일 토픽을 선택할 수 있습니다.

**3-2) 선택한 토픽 T에서 단어의 출현 확률 분포에 기반해 문서에 사용할 단어를 고릅니다.**
- Ex) 강아지 토픽을 선택하였다면, 33% 확률로 강아지란 단어를 선택할 수 있습니다. 3)을 반복하면서 문서를 완성합니다.

이러한 과정을 통해 문서가 작성되었다는 가정 하에 LDA는 토픽을 뽑아내기 위하여 위 과정을 역으로 추적하는 역공학(reverse engneering)을 수행

## 1.1 **잠재 디리클레 할당과 잠재 의미 분석의 차이**

<aside>
👌🏻 **LSA** : DTM을 차원 축소 하여 축소 차원에서 근접 단어들을 토픽으로 묶는다.

**LDA :** 단어가 특정 토픽에 존재할 확률과 문서에 특정 토픽이 존재할 확률을 결합 확률로 추정하여 토픽을 추출한다.

</aside>

## 1.2 LDA 특징

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5ce5661c-e694-4245-b7c9-156d5758c81c/Untitled.png)

- 디리클레(Dirichlet)의 분포를 가정하고 토픽이라는 잠재(Latent)된 변수를 활용하여, 각 문서들과 단어들이 토픽에 할당되는 확률 분포를 그린 것.
- 디리클레의 분포는 확률분포 중 다항분포, 연속확률분포로 중 하나로 정의할 수 있고, 디리클레의 분포의 성질 중 하나는 k차원의 실수 벡터를 모두 더한 값은 1이다. 
(토픽의 단어를 모두 더한 값은 1, 토픽들의 요소 값을 모두 더한 값은 1로 정의)
- “사전 켤레 확률” 의 성질을 가지고 있다

→ 디리클레분포(Dirichlet Distribution) : 이항분포가 아닌, 연속확률 분포 중 하나로 k차원의 실수 벡터 중 벡터의 요소가 양수이며 모든 요소를 더한 값을 1로 하여 확률 값이 정의되는 분포

# 2. LDA 수행과정

**1) 사용자는 알고리즘에게 토픽의 개수 k를 알려줌.**

→ 앞 LDA는 토픽의 개수 k를 입력받으면, k개의 토픽이 M개의 전체 문서에 걸쳐 분포되어 있다고 가정

**2) 모든 단어를 k개 중 하나의 토픽에 할당.**

**→**이제 LDA는 모든 문서의 모든 단어에 대해서 k개 중 하나의 토픽을 랜덤으로 할당

이 작업이 끝나면 각 문서는 토픽을 가지며, 토픽은 단어 분포를 가지는 상태 
⇒ 랜덤으로 할당하였기 때문에 사실 이 결과는 전부 틀린 상태

만약 한 단어가 한 문서에서 2회 이상 등장하였다면, 각 단어는 서로 다른 토픽에 할당되었을 수도 있다.

**3) 이제 모든 문서의 모든 단어에 대해서 아래의 사항을 반복 진행 (iterative)**

**3-1) 어떤 문서의 각 단어 w는 자신은 잘못된 토픽에 할당되어져 있지만, 다른 단어들은 전부 올바른 토픽에 할당되어져 있는 상태라고 가정.**

→ **이에 따라 단어 w는 아래의 두 가지 기준에 따라서 토픽이 재할당**

- p(topic t | document d) : 문서 d의 단어들 중 토픽 t에 해당하는 단어들의 비율

- p(word w | topic t) : 각 토픽들 t에서 해당 단어 w의 분포

이를 반복하면, 모든 할당이 완료된 수렴 상태가 됨.

![https://wikidocs.net/images/page/30708/lda1.PNG](https://wikidocs.net/images/page/30708/lda1.PNG)

- 위의 그림은 두 개의 문서 doc1과 doc2가 존재, doc1의 세번째 단어 apple의 토픽을 결정하고자 함

![https://wikidocs.net/images/page/30708/lda3.PNG](https://wikidocs.net/images/page/30708/lda3.PNG)

- 첫번째로 사용하는 기준은 문서 doc1의 단어들이 어떤 토픽에 해당하는지를 판단.
- doc1의 모든 단어들은 토픽 A와 토픽 B에 50 대 50의 비율로 할당되어져 있으므로, 이 기준에 따르면 단어 apple은 토픽 A 또는 토픽 B 둘 중 어디에도 속할 가능성이 있음.

![https://wikidocs.net/images/page/30708/lda2.PNG](https://wikidocs.net/images/page/30708/lda2.PNG)

두번째 기준은 단어 apple이 전체 문서에서 어떤 토픽에 할당되어져 있는지를 보고, 기준에 따라 단어 apple은 토픽 B에 할당될 가능성이 높음.

→ 이러한 두 가지 기준을 참고하여 LDA는 doc1의 apple을 어떤 토픽에 할당할지 결정

# 3. 한국해양개발원 뉴스 LDA 토픽모델링

## 3.1. MongoDB 구축

- **Version**

<aside>
⭐ **Centos 7
Python 3.6
mongoDB 4.4
chromeDriver 104.0.5112.xx**

</aside>

- 기본 명령어(생성, 확인)

```bash
# 간단한 명령어
mongod --version # 명령어 확인
use test # test DB 이동
show dbs # 데이터베이스 목록 확인
db.stats # db 상태확인
--------------------------------------
#데이터베이스 shutdown
use admin
db.shutdownServer()
#logout
db.logout()
#collection create
db.createCollection("{NAME}")
```

### 3.1.1. Mongo 설정

- **권한 설정**

```bash
# 권한설정
vi /etc/mongod.conf

''sh 
bindip : 127.0.0.1 -> 0.0.0.0 # 외부접속 허용

#security:
#  authorization : enable
''
```

- **인증 추가**

```bash
#인증 추가
db.createUser({ user:"{id}", pwd: "{password}", roles:[{ role: "dbOwner", db: "{DBName}"}] }]
```

- **Mongo DB path 수정**

```bash
#mongoDB에서 기본데이터디렉토리가 /var/lib/mongodb로 잡혀있음
#mongodb 디렉토리 생성

sudo mkdir /data && sudo mkdir /data/mongodb

#db를 중지시킨뒤 이전할 디렉토리에 권한을 부여
service mongodb stop
chown mongodb:mongodb /data/mongodb/

#데이터를 복사 & 이전
cp -arp /var/lib/mongodb/* /data/mongodb

```

- **config 파일 수정**

```bash
#컨피그 파일 수정
vim /etc/mongodmon.conf

''sh
dbpath=/data/mongodb  ### db데이터 폴더위치 변경
''
```

## 3.2. Mongo DB data input

<aside>
💡 **해양, 수산, 항만, 해운 분야 뉴스 18-21년까지의 데이터 약 110만건의 데이터를 몽고 서버에 넣고 특정 키워드가 포함된 데이터를 검색**

</aside>

- **Mongo Import 사용코드**

```bash
# authentication + mongo import
mongoimport --db marine --collection news \
          --authenticationDatabase admin --username <user> --password <password> \
          --drop --file news.json
```

- Data 조회(Mongo import 성공 화면)

```sql
db.getCollection('marine').find({}).count()
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dbf74117-fea6-42b4-bff2-de4a4f0f6264/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c1f84dfb-5d07-4276-9d02-1af8ce8fbfff/Untitled.png)

### 3.2.1. 에러메세지(해결 과정)

- **MongoDB import Error**

<aside>
💡 >> **Failed: error processing document #214167: invalid character 's' after object key:value pair imported 214152 documents
#** json 파일 내 214167번째 줄에 중복된 KEY:PAIR 있음.

</aside>

2개의 뉴스가 구분자 없이 붙어있었며, json 파일 해당 뉴스 삭제 후 재 임포트 진행

- **Text Search 에러**

```sql
**## OperationFailure: text index required for $text query**
db.reviews.createIndex( { comments: "text" } )
```

Text 인덱스를 만들어 줌으로써 해결

- **Mongo restart Error**

<aside>
💡 **>> Mongo DB restart fails. Log says that service failed to start with return exit-code error of 14.**

</aside>

```sql
# Centos7 mongodb 4.4 설치시 selinux 및 모든 권한 완료 후 error code 14 발생

$ rm -rf mongod:mongod /tmp/mongodb-27017.sock #소켓 삭제
$ sudo systemctl start mongod
```

참고사이트 : /https://hoing.io/archives/1102

### 3.2.2. $TEXT검색을 위한 PyMongo 활트

```python
import pymongo

#mongodb 연결
conn = pymongo.MongoClient("mongodb://localhost:27017/")

#db&collection
mydb = conn['news']
myColl = mydb['marine']

#query
key={'({$text:{$search: "조선 해운 해양 수산" }})'}
marine_search = myColl.find(key)

# mongodb aggregate 활용
db.marine.aggregate(
    [
        { $match: { $text:{ $search: "조선 해운 해양 조선업 해운업 수산업 어촌 해상" } } },
        { $group: { _id: "$service_type", count: {$sum: 1} } }
    ]
)
```

## 3.2. 데이터 불러오기

- 해양, 수산, 항만, 해운 분야 뉴스 22년 데이터 1.4만건의 데이터가 담긴 MongoDB를 연결.

```python
# import
import pandas as pd
import re
import pymongo
from konlpy.tag import Mecab
from tqdm.auto import tqdm

# MONGO DB connection
host = '172.30.1.30'
port = 27017

conn = pymongo.MongoClient(host=host, port=port)
db = conn.get_database('news')  ## db name
collection = db.get_collection('marine22')
```

- $text search를 이용해 해운, 항만, 조선, 해양 키워드를 추출

```python
# Mongo search
news_marine = list(collection.find({'service_type':'NEWS', '$text':{'$search' : '해운 항만 조선 해양'}}, {'_id':0, 'content':1}))
news_content = [news['content'] for news in news_marine]
news_data = pd.DataFrame(news_content)
print('MongoDB Connected.')
# conn close
conn.close()
print('MongoDB Closed.')

# 데이터 확인
news_data
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/56771b2c-003e-4ab0-aabf-412baf388a95/Untitled.png)

## 3.3. 텍스트 데이터 전처리

- **전처리 함수**

```python
# 한글, 영문, 0-9 숫자 빼고 삭제
def clean_text(text):
    text = text.replace(".", "").strip()
    text = text.replace("·", " ").strip()
    pattern = '[^ ㄱ-ㅣ가-힣|0-9|a-zA-Z]+'
    text = re.sub(pattern=pattern, repl='', string=text)
    return text
# 가져올 단어 선택
def get_nouns(tokenizer, sentence):
    tagged = tokenizer.pos(sentence)
    # 한개의 단어 삭제 포함
    nouns = [s for s, t in tagged if t in ['NNG', 'NNP', 'SL', 'XR'] and len(s) >1]
    return nouns

# 토크나이저는 Mecab으로(okt가 편리하나 Mecab의 속도가 가장 빨랐다.)
def tokenize(data):
    tokenizer = Mecab(dicpath='/Users/yhkoo/mecab-ko-dic-2.0.3-20170922')
    processed_data = []
    for sent in tqdm(data):
        sentence = clean_text(str(sent).replace("\n", "").strip())
        processed_data.append(get_nouns(tokenizer, sentence))
    return processed_data

# 처리
processed_data = tokenize(newsdata['content'])
```

## 3.4. LDA 학습

- **LDA 사용을 위해 Gensim을 임포트**

```python
from gensim.models.ldamodel import LdaModel
from gensim.models.callbacks import CoherenceMetric
from gensim import corpora
from gensim.models.callbacks import PerplexityMetric
from gensim.models.coherencemodel import CoherenceModel

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```

### 3.4.1. **corpora.Dictionary()**

- 각 단어에 정수 인코딩을 하는 동시에, 각 뉴스에서의 단어의 빈도수를 기록
- 여기서는 각 단어를 (word_id, word_frequency)의 형태로 바꿈.
- word_id는 단어가 정수 인코딩된 값이고, word_frequency는 해당 뉴스에서의 해당 단어의 빈도수를 의미

```python
# 데이터를 dictionary형태로 명사 리스트 만들기
dictionary = corpora.Dictionary(processed_data)

# 명사 형태의 문서별로 말뭉치 만들기
corpus = [dictionary.doc2bow(text) for text in processed_data]
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/56f846f6-ceba-47a8-a158-e3a9607f6436/Untitled.png)

### 3.4.2. CoherenceModel 을 통한 토픽 최적화

1. 의미 : 토픽이 얼마나 의미론적으로 일관성 있는지 판단. 높을수록 의미론적 일관성 높음
2. 주 용도 : 해당 모델이 얼마나 실제로 의미 있는 결과를 내는지 확인
3. 기존에 언어모델 평가로 CoherenceModel만을 사용 후 원하는 토픽 개수의 Coherence 모델을 지속 학습 시켜 토픽을 할당 했으나, 추후에는 두 가지 모델 함께 적용 해보는 것과 좀 더 정밀한 사용이 필요
- **Coherence Model**

```python
coherence_values=[]
for i in range(2,15):
    ldamodel = LdaModel(corpus, num_topics=i, id2word=dictionary)
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=processed_data, dictionary=dictionary, topn=10)
    coherence_lda = coherence_model_lda.get_coherence()
    coherence_values.append(coherence_lda)
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4671fe4c-48a7-49bb-9af4-4ad414e816fa/Untitled.png)

- **Coherence Graph**

```python
import matplotlib.pyplot as plt
x = range(2,15)
plt.plot(x, coherence_values)
plt.xlabel('number of topics')
plt.ylabel('coherence score')
plt.show()
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/89f836a8-766f-4990-9672-6b34741bd494/Untitled.png)

### 3.4.3. 퍼플렉서티(perplexity) : 
           PPL로 줄여서 표현 : ‘perplexed : 헷갈리는‘ 과 유사한 의미

1. 선정된 토픽 개수마다 학습시켜 가장 낮은 값을 보이는 구간을 찾아 최적화된 토픽의 개수 선정 가능
2. 의미 : 확률 모델이 결과를 얼마나 정확하게 예측하는지 판단. → 낮을수록 정확하게 예측.
3. 주 용도 : 동일 모델 내 파라미터에 따른 성능 평가할 때 주로 사용
4. 한계 : Perplexity가 낮다고 해서, 결과가 해석 용이하다는 의미가 아님
- **Perplexity Model**

```python
perplexity_values= []
for i in range(2,20):
    ldamodel = LdaModel(corpus, num_topics=i, id2word=dictionary)
    perplexity_values.append(ldamodel.log_perplexity(corpus))
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/27c88173-039c-48b2-84e3-d9d26957df9b/Untitled.png)

- **Perplexity Graph**

```python
x=range(2,20)
plt.plot(x, perplexity_values)
plt.xlabel('number of topics')
plt.ylabel('perplexity score')
plt.show()
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cfa21daa-296c-47fd-9b3e-79fdedc082e3/Untitled.png)

- 언어모델 평가로 CoherenceModel만을 사용 후 원하는 토픽 개수의 Coherence 모델을 지속 학습 시켜 토픽을 할당 했으나, 추후에는 두 가지 모델 함께 적용 해보는 것과 좀 더 정밀한 사용이 필요

```python
ldamodel = LdaModel(corpus, num_topics=12 , alpha=0.01, id2word=dictionary)
result = ldamodel.print_topics(num_words=12) # num_words=12로 총 12개의 단어만 출력 
result
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/46920d18-f498-4096-8fd8-35409bf78d0c/Untitled.png)

→ 각 단어 앞에 붙은 수치는 단어의 해당 토픽에 대한 기여도를 뜻함

### 3.4.4. 토픽에 할당된 문서 추출(Topic-Document)

```python
def make_topictable_per_doc(ldamodel, corpus):
    topic_table = pd.DataFrame()
    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 꺼내옴
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list
        doc = sorted(doc, key=lambda x : (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽 순으로 토픽을 정렬
        # 문서 0번 (1 30, 2 15, 3 10, 4 12.5) -> (1 30, 2 15, 4 12.5, 3 10)

        # 모든 문서에 대해 각각 아래 수행
        for j, (topic_num, prop_topic) in enumerate(doc): # 몇 번 토픽인지와 비중을 나누어 저장
            if j == 0: # 정렬했으므로 가장 앞에 있는 것이 가장 비중이 높음
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True) # 가장 비중이 높은 토픽, 가장 비중이 높은 토픽의 비중, 전체 토픽의 비중을 저장
            else:
                break
    return(topic_table)
```

- 각문서별로토픽에할당되는토픽번호와차지하는비중을만들기위한코드
- 해당코드를통해문서개별로가장크게할당된토픽의번호와비율확인가능, 여러 토픽에 중첩 할당 된 경우, 개별 할당된 값도 확인 가능

```python
topictable = make_topictable_per_doc(ldamodel, corpus)
topictable = topictable.reset_index() # 문서 번호를 의미하는 열로 사용하기 위해 인덱스 열을 만
topictable.columns=['문서번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중','각 토픽의 비중']
topictable[:10]
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/aacb3890-28e8-410a-b89f-0807238640cb/Untitled.png)

- [분석 과정에서의 Idea]
    
    단어별로 토픽 모델링 결과를 잘 나타내고서, 문서별로 다시 묶은 다음 각 주제에 해당하는 문서들끼리만 다시 토픽모델링 결과를 낸다면 하나의 주제에서 또 다르게 얘기하는 주제들을 끄집어 낼 수 있음.
    

### 3.4.5 ****LDA 시각화 하기****

- PyLDAvis 사용

```python
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
prepared_data = gensimvis.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(prepared_data)
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7de8b9c9-521e-48d9-aa1f-7315c4c333b8/Untitled.png)

# & 참조

1. [https://wikidocs.net/30708](https://wikidocs.net/30708)
2. 엠포스-데이터랩_토픽모델링LDA방법론-정리.pdf
