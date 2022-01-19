"""
*.xml -> *.txt  format = sentence + '\t' + emotion_type + '\n'
"""



import xml.dom.minidom as xmldom
import os
import random
from config import SEED

random.seed(SEED)


xmlfilepath_1 = os.path.join('..', 'data', 'nlpcc2014', 'Training data for Emotion Classification.xml')
xmlfilepath_2 = os.path.join('..', 'data', 'nlpcc2014', 'EmotionClassficationTest.xml')
xmlfilepath_3 = os.path.join('..', 'data', 'nlpcc2013', '微博情绪样例数据V5-13.xml')
xmlfilepath_4 = os.path.join('..', 'data', 'nlpcc2013', '微博情绪标注语料.xml')
xmlfilepath_5 = os.path.join('..', 'data', 'nlpcc2014', 'NLPCC2014微博情绪分析样例数据.xml')

data = {}
emotion_num = {'anger': 0,
               'disgust': 0,
               'fear': 0,
               'happiness': 0,
               'like': 0,
               'sadness': 0,
               'surprise': 0,
               'other': 0}
emotion_map = {'愤怒': 'anger',
               '厌恶': 'disgust',
               '恐惧': 'fear',
               '高兴': 'happiness',
               '喜好': 'like',
               '悲伤': 'sadness',
               '惊讶': 'surprise'}

for xmlfilepath in [xmlfilepath_1, xmlfilepath_2, xmlfilepath_3, xmlfilepath_4, xmlfilepath_5]:

    print('Start parsing ', xmlfilepath)
    
    domobj = xmldom.parse(xmlfilepath)
    
    elementobj = domobj.documentElement
    
    subElementObj = elementobj.getElementsByTagName("sentence")

    print('There are ' + str(len(subElementObj)) + ' sentences in current xml file')
    
    for sentence in subElementObj:
        text = sentence.firstChild.data

        if text in list(data.keys()) or len(text) < 4:
            pass
        else:
            emotion_tag = sentence.getAttribute('opinionated')
            if not emotion_tag:
                emotion_tag = sentence.getAttribute('emotion_tag')

            assert emotion_tag in ['Y', 'N']

            if emotion_tag == 'N':
                if random.random() < 0:
                    continue
                else:
                    label = 'other'
            elif emotion_tag == 'Y':
                label = sentence.getAttribute('emotion-1-type')
                if label not in emotion_num.keys():
                    label = emotion_map[label]

                assert label in emotion_num.keys()

            data[text] = label
            emotion_num[label] = emotion_num[label] + 1
    print('There are ' + str(len(data)) + ' sentences in total')

print('saving.....')
out = open(os.path.join('..', 'data', 'raw_data.txt'), 'w')
for text, label in data.items():
    out.write(text + '\t' + label + '\n')
print('success saving to data/raw_data.txt')

print('sentences: ', len(data))
for label, num in emotion_num.items():
    print(label, ': ', num, ' ', num / len(data) * 100, '%')
