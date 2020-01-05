############################################  NOTE  ########################################################
#
#           Creates NER training data in Spacy format from JSON downloaded from Dataturks.
#
#           Outputs the Spacy training data which can be used for Spacy training.
#
############################################################################################################
import json
import random
import logging
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_scor

def check_spacy():
    print('*****************************************\n')
    nlp = spacy.load(r'D:\PyCharm\training_ner\model_1_ner')
    text1 = u'''3-D, Chocolate Factory ink
'''
    textList = [
        u'''Iris Bohnet, Member of the Board of Directors
        ''',
        u'''USA 444 N. Michigan Ave.,Suite 2600 Chicago,Illinois
        ''',
        u'''60611 +1-312-704-5100,info@thesciongroup.com''',
        u'''Intellect Design Arena Limited''',
        u'''Apache Corporation''',
        u'''Archer Daniels Midland''',
        u'''Miss Bailey Brown Chocolate Corporation''',
        u'''Kims Chocolate NV''',
        u'''The Belgian Chocolate Group''',
        u'''Chocolate Cafe Franchsing Shop''',
        u'''Scion''',
        u'''1832 Asset Management''',
        u'''Woori Bank'''
    ]
    doc_to_testList = []

    entitiesList = []
    j = 1
    for i in textList:
        entities = {}
        testList_nlp = nlp(i)
        for ent in testList_nlp.ents:
            entities[ent.label_] = ent.text
        doc_to_testList.insert(j,testList_nlp)

        entitiesList.insert(j,entities)
        j = j + 1
    print(entitiesList)
    #--------------------------
    # doc_to_test1 = nlp(text1)
    # print("After NLP",doc_to_test1)
    # entities = {}
    # for ent in doc_to_test1.ents:
    #     entities[ent.label_] = ent.text
    #
    # print(entities)

def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines=[]
        with open(dataturks_JSON_FilePath, 'r', encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                #print('labels', labels)
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    entities.append((point['start'], point['end'] + 1 ,label))
                    #print('label is', label)

            training_data.append((text, {"entities" : entities}))

        return training_data
    except Exception as e:
        print(line)
        logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None

################### Train Spacy NER.###########
def train_spacy():

    TRAIN_DATA = convert_dataturks_to_spacy(r"D:\PyCharm\training_ner\Totraindata.json")
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    print('TRAIN_DATA...', TRAIN_DATA)
    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])
            # print("added label is",ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(50):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    #test the model and evaluate it
    examples = convert_dataturks_to_spacy(r"D:\PyCharm\training_ner\Totraindata.json")
    tp=0
    tr=0
    tf=0

    ta=0
    c=0
    for text,annot in examples:

        f=open("invoice"+str(c)+".txt","w")
        doc_to_test=nlp(text)
        d={}
        for ent in doc_to_test.ents:
            d[ent.label_]=[]
            print("lab is", ent.label_)
        for ent in doc_to_test.ents:
            d[ent.label_].append(ent.text)
            print("text is",ent.text)
        for i in set(d.keys()):

            f.write("\n\n")
            f.write(i +":"+"\n")
            for j in set(d[i]):
                f.write(j.replace('\n','')+"\n")
        d={}
        for ent in doc_to_test.ents:
            d[ent.label_]=[0,0,0,0,0,0]
        for ent in doc_to_test.ents:
            doc_gold_text= nlp.make_doc(text)
            gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
            y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
            y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in doc_to_test]
            if(d[ent.label_][0]==0):
                #f.write("For Entity "+ent.label_+"\n")
                #f.write(classification_report(y_true, y_pred)+"\n")
                (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
                a=accuracy_score(y_true,y_pred)
                d[ent.label_][0]=1
                d[ent.label_][1]+=p
                d[ent.label_][2]+=r
                d[ent.label_][3]+=f
                d[ent.label_][4]+=a
                d[ent.label_][5]+=1
        c+=1
    for i in d:
        print("\n For Entity "+i+"\n")
        print("Accuracy : "+str((d[i][4]/d[i][5])*100)+"%")
        print("Precision : "+str(d[i][1]/d[i][5]))
        print("Recall : "+str(d[i][2]/d[i][5]))
        print("F-score : "+str(d[i][3]/d[i][5]))

    nlp.to_disk(r'D:\PyCharm\training_ner\model_1_ner')
    check_spacy()

train_spacy()
