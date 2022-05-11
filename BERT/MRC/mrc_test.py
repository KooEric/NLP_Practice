from transformers import pipeline
question_answering = pipeline("question-answering", model="ainize/klue-bert-base-mrc", tokenizer="ainize/klue-bert-base-mrc")
content = open('content_utf8.txt', 'rt', encoding='utf-8').read()
print("Content\n" + content)
question = '자택과 집무실 간 이동 거리는 얼마인가?'
print("Question:\n" + question)
result = question_answering(question=question, context=content)
print("Answer:{}, Score: {}".format(result['answer'], round(result['score'],3)))

#Answer
'''
Question:
자택과 집무실 간 이동 거리는 얼마인가?

Answer:
약 7㎞다, Score: 0.897
'''