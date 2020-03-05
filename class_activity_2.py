import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import pandas as pd
import spacy
import re
from spacy import displacy
from nltk.tag import StanfordNERTagger


def lemmatize_spacy(article):
    nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
    text = nlp(article)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    print(text)



def lemmatize_nltk(article):
    #nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

    pairs = nltk.pos_tag(nltk.word_tokenize(article))
    return_list = list()
    for one in pairs:
        return_list.append((one[0], ":", lemmatizer.lemmatize(one[0])))


def pos_spacy(article):
    nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
    sentence_nlp = nlp(article)
    spacy_pos_tagged = [(word, word.tag_, word.pos_) for word in sentence_nlp]
    pos = pd.DataFrame(spacy_pos_tagged, columns=['Word', 'POS tag', 'Tag type']).T
    print(pos)


def pos_nltk(article):
    nltk.download('averaged_perceptron_tagger')
    nltk_pos_tagged = nltk.pos_tag(nltk.word_tokenize(article))
    pos = pd.DataFrame(nltk_pos_tagged, columns=['Word', 'POS tag']).T
    print(pos)


def tokenize_nltk(text):
    # nltk.download('punkt')
    tokenized = word_tokenize(text)
    print(tokenized)


def tokenize_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    token_list = list()
    for token in doc:
        token_list.append(token.text)
    print(token_list)


def get_text():

    url = 'http://www.oudaily.com/news/ou-s-italy-coronavirus-closures-bring-student-' \
          'disappointment-dean-stresses/article_5596b268-5d05-11ea-8ebb-a3d2d02f064e.html'
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    text = soup.select(".subscriber-premium")

    article = [t.text for t in text]



    article = article[0]

    article = article.replace("TNCMS.AdManager.render({region: 'fixed-big-ad-top-asset', slot: 1, fold: \"above\"});", "")
    article = article.replace("TNCMS.AdManager.render({region: 'fixed-big-ad-middle-asset', slot: 1, fold: \"span\"});", "")
    article = article.replace("TNCMS.AdManager.render({region: 'fixed-big-ad-bottom-asset', slot: 1, fold: \"below\"});", "")

    return(article)


def entity_recognition_spacy(article):
    nlp = spacy.load('en_core_web_sm')
    text_nlp = nlp(article)
    ner_tagged = [(word.text, word.ent_type_) for word in text_nlp]
    print(ner_tagged)
    named_entities = []
    temp_entity_name = ""
    temp_named_entity = None
    for term, tag in ner_tagged:
        if tag:
            temp_entity_name = ' '.join([temp_entity_name, term]).strip()
            temp_named_entity = (temp_entity_name, tag)
        else:
            if temp_named_entity:
                named_entities.append(temp_named_entity)
                temp_entity_name = ""
                temp_named_entity = None
    print(named_entities)


def entity_recognition_nltk(article):
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    ne = nltk.ne_chunk(nltk.pos_tag(word_tokenize(article)))
    print(ne)


def dependency_spacy(article):
    nlp = spacy.load('en_core_web_sm')
    text_nlp = nlp(article)
    print(displacy.render(text_nlp))

def dependency_nltk(article):
    from nltk.parse.stanford import StanfordDependencyParser
    #sdp = StanfordDependencyParser(path_to_jar='E:/stanford/stanford-parserfull-2015-04-20/stanford-parser.jar',
    #                               path_to_models_jar='E:/stanford/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar')
    #result = list(sdp.raw_parse(sentence))[0]
    #print(result)


if __name__ == "__main__":

    article = get_text()

    tokenize_nltk(article)
    tokenize_spacy(article)

    pos_spacy(article)
    pos_nltk(article)

    lemmatize_spacy(article)
    lemmatize_nltk(article)

    entity_recognition_spacy(article)
    entity_recognition_nltk(article)

    dependency_spacy(article)
    dependency_nltk(article)
