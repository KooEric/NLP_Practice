# Making Perfume Name Entities Recongnition Spacy Model
### Rule-based Model

1. Crawling Perfume Reviews and Details
2. Make Perfume Note, Accord Entities


# 1.  문제 정의

<aside>
💡 데이터 바우처 이후 **향수 추천 서비스를 하기 위해** **향과 노트, 노트 조합등의 개체명 인식이 필요하므로,** 개체명 인식기를 개발하였음.

</aside>

## 1.1 Spacy Rule-based Model 사용 이유

딥러닝 모델을 사용하기 위해 가장 중요한 데이터셋을 만드는 것이 중요한데, NER dataset의 경우 공유된 dataset이 많지 않으며, 구축하기 까다롭다. 

데이터셋을 구축하여 해당 모델에 학습까지의 시간이 촉박하여 해당 인식기는 흔히 사용하는 딥러닝 모델인 BERT-NER 또는 LSTM+CRF이 아니라, 영어 전처리 라이브러리인 Spacy를 이용하여 Rule-based 모델을 제작하였음.

# 2. 수집 범위 및 데이터 확인

## 2.1 수집범위

수집 데이터는 [https://www.fragrantica.com/](https://www.fragrantica.com/) 사이트의 향수 정보임.

수집된 데이터는 엑셀 형식으로 perfumeDetail_result.csv, perfumeReview_result.csv 상세정보와 리뷰데이터 두 개의 파일로 구성되어 있다.

## 2.2 데이터 확인

- Detail 파일(63,486개)

해당 모델에서는 accords와 notes 정보만을 사용함.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/02e377f9-f3d6-44d9-ba96-8915ddbf0e94/Untitled.png)

- Review 파일(398,583개)

해당 모델에서는 review 데이터만 사용함.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8b2f18b4-04c8-4780-9c3a-6401e0c66874/Untitled.png)

# 3. 코드 내용

<aside>
🖋️ Spacy 모델의 경우, 3.0 버전이 새로 업데이트 되어 기존 2.1 버전과 다른 부분이 있어, 3.0버전에 맞춰 제작하였음.

</aside>

## 3.1 Import Part

```python
import pandas as pd # csv 파일
from nltk.corpus import stopwords # 영어 전처리
from nltk.tokenize import RegexpTokenizer # 영어 전처리
from __future__ import unicode_literals, print_function
import random
from pathlib import Path
import spacy # 개체명 학습
from tqdm import tqdm # 학습과정 확인
from spacy.util import minibatch # 개체명 학습
from spacy.training.example import Example # 개체명 학습
```

## 3.2 Detail 데이터 패턴화 작업

- Detail 정보 판다스 프레임화

```python
nlp = spacy.load('en_core_web_sm')
df = pd.read_csv('/saltlux_BERT/perfumeReview_crawling/perfumeDetail_result.csv', \n
									encoding='utf-8').fillna("") # fillna("")를 통해 결측값의 공백을 메워줌
df.head()
```

spacy.load(’en_core_web_sm’)의 에러가 발생할 경우, spacy.download()를 통해 다운로드하여야 함.

```python
df['notes']=df['notes'].dropna() # notes 컬럼 결측값 삭제
df['accords']=df['accords'].dropna() # accords 컬럼 결측값 삭제
```

- note, accord 정보 가져오기

Detail 파일 내 notes, accords는 ‘|’와 같은 구분자로 수집되어 있음.

```python

# 빈 리스트에 뭉쳐있는 값들을 하나씩 만들어준다.
## accords
accords = df['accords'].tolist()
accord = []

for i in range(len(accords)):
    tmp = accords[i].split('|')
    for j in tmp:
        if j not in accord:
            accord.append(j)
## notes
notes = df['notes'].tolist()
note = [] 

for i in range(len(notes)):
    tmp = notes[i].split('|')
    for j in tmp:
        if j not in note:
            note.append(j)
```

- 빈 값들로 인한 사소한 에러들 사전 제거

```python
note.remove('')
accord.remove('')
```

- 해당 데이터들을 패턴화 하기 위한 작업

```python
accord = set(accord)
note = set(note)
```

```python
accord_patterns = []
for item in note:
    accord_pattern = {'label': 'accord', 'pattern': item}
    accord_patterns.append(accord_pattern)
accord_patterns

note_patterns =[]
for item in note:
    note_pattern = {'label': 'note', 'pattern': item}
    note_patterns.append(note_pattern)
note_patterns
```

- 패턴화된 값들을 하나의 통합 패턴으로 만들기

```python
total_patterns = []
total_patterns.extend(note_patterns)
total_patterns.extend(accord_patterns) # 리스트 확장

total_patterns[1000] # 데이터 확인
```

결과값 : {'label': 'note', 'pattern': 'Pitanga'}

## 3.3 리뷰 데이터 전처리

```python
df2 = pd.read_csv('/saltlux_BERT/perfumeReview_crawling/perfumeReview_result.csv', \n
									encoding='utf-8').fillna("") # 리뷰데이터 불러오기
```

```python
# 함수 선언
def make_lower_case(text): # 소문자로 만들어준다.
    return text.lower()

def remove_stop_words(text): # Stop_word(ex. the, a 등을 삭제)
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text
```

- 전처리 함수 적용

```python
# df2['review'] = df2.review.apply(func=make_lower_case)
df2['review'] = df2.review.apply(func=remove_stop_words)

rr = df2['review'].tolist() # 리뷰데이터 리스트화
```

해당 부분에선 고유의 향을 적용하기 위해 소문자 함수를 적용하지 않았다.

- 학습 데이터 수량 설정

```python
total_R = ','.join(rr[:2000]) # 메모리 초과로 인해 2000건의 데이터로 학습을 진행
total_R
```

## 3.4 Rule-based 모델 학습

- 학습 데이터 만들기

```python
# 학습
nlp = spacy.load("en_core_web_sm")
text = total_R
corpus = []

doc = nlp(text)
for sent in doc.sents:
    corpus.append(sent.text)

nlp = spacy.blank("en")
ruler = nlp.add_pipe("entity_ruler")
patterns = total_patterns
# nlp = spacy.blank("en")
ruler.add_patterns(patterns)

TRAIN_DATA = []
for sentence in corpus:
    doc = nlp(sentence)
    entities = []

    for ent in doc.ents:
        if len(ent) > 0:
            entities.append((ent.start_char, ent.end_char, ent.label_))
        # entities.append((ent.text, ent.label_))
        TRAIN_DATA.append((sentence, {"entities": entities}))

print (TRAIN_DATA)
```

```python
# 학습
model = None
output_dir=Path("./content")
n_iter=50

#load the model
if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')
    print("Created blank 'en' model")

#set up the pipeline

if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner', last=True)
else:
    ner = nlp.get_pipe('ner')
```

- 모델 학습

```python
# 학습 시작
from spacy.training.example import Example

for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

loss_history = []
train_examples = []
for text, annotations in TRAIN_DATA:
    train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(train_examples, size=spacy.util.compounding(4.0, 32.0, 1.001))
        batches_list = [(idx, batch) for idx, batch in enumerate(batches)]
        for idx, batch in tqdm(batches_list):
            nlp.update(
                batch,
                drop=0.5,
                sgd=optimizer,
                losses=losses)
        loss_history.append(losses)
        print(losses)
```

- 학습 결과 확인

```python
for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
```

- 학습 모델 저장

```python
# 학습
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
```

- 모델 전체 리뷰데이터 적용

```python
# 저장된 개체명 인식 모델 불러오기, 활용
print("Loading from", '/content')
nlp2 = spacy.load('/Users/yhkoo/PycharmProjects/pythonProject/deep/saltlux_BERT/content')
DATA = df2['review']
dataframe = []
for text in DATA:
    if len(ent) > 0:
        doc = nlp2(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    dataframe.append({'Entities': [(ent.text, ent.label_) for ent in doc.ents]})
```

## 3.5 최종 결과 통합

```python
result =pd.DataFrame(dataframe, columns=['Entities']) # Entities 컬럼명으로 데이터 프레임화

final = pd.concat([df2, result], axis=1) # 리뷰 데이터와 결함
final.to_csv('./final_merge_entitiy.csv') # csv파일 만들기
```
