from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pymysql

default_args = {
        'start_date':datetime(2022, 10, 9),
        'owner': 'airflow',
        'schedule_interval':'@daily',
        'catchup':True
        }
conn = pymysql.connect(
        user = 'sortness',
        password = 'sortness1!',
        host = 'database-1.ckt7rdbj2s8a.us-west-1.rds.amazonaws.com',
        db = 'articles',
        charset = 'utf8'
        )

from bs4 import BeautifulSoup
import re

def crawling_inside(html, ranking_date, url, cursor):
    soup = BeautifulSoup(html, 'html.parser')
    article_date, section, newspaper_com, title, content = '', '', '', '', ''
    try:
        article_date = soup.find("span", "media_end_head_info_datestamp_time _ARTICLE_DATE_TIME").get_text()
        article_date = article_date[:10].replace(".", "-")

        ranking_date = ranking_date[0:4]+'-'+ranking_date[4:6]+'-'+ranking_date[6:]
        section_button = soup.find("li", class_ = 'Nlist_item _LNB_ITEM is_active')
        
        if section_button:
            section = section_button.get_text()
        else:
            return None

        newspaper_com = soup.find("a", class_ = "ofhd_float_title_text").get_text()
        if newspaper_com == "코리아헤럴드" or newspaper_com == "코리아중앙데일리":
            return None

        title = soup.h2.get_text()
        print(title, article_date)

        content = soup.find("div", id = "dic_area").get_text()
        content = re.sub(r'[\n\r\t"<br>"]', '', content)

        sql = "insert into article(ranking_date, article_date, section, newspaper_com, title, content, link) values(%s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(sql, (ranking_date, article_date, section, newspaper_com, title, content, url))
    
    except:
        return None 

from bs4 import BeautifulSoup
import csv
import time
import urllib.request
import pymysql
from urllib.error import HTTPError

def crawling_main(**context):
    ranking_date = str(context['execution_date'])[:10].replace('-', '')
    print(ranking_date)
    url = "https://news.naver.com/main/ranking/popularDay.naver?date=" + ranking_date
    source_code_from_URL = urllib.request.urlopen(url)

    soup = BeautifulSoup(source_code_from_URL, "html.parser")

    tag_name = "list_title nclicks('RBP.rnknws')"
    tag_list = soup.find_all("a", tag_name)
    
    cursor = conn.cursor()
    for tag in tag_list:
        try:
            source = urllib.request.urlopen(tag['href'])
            crawling_inside(source, ranking_date, tag['href'], cursor)
        except HTTPError as e:
            err = e.read()
            code = e.getcode()
            print(code)
    
    conn.commit()
    conn.close()

import pandas as pd
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import NounMatchTokenizer
from konlpy.tag import Okt
from numpy import empty
from tempfile import NamedTemporaryFile
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from io import BytesIO

def get_s3_stopwords(filename):
    hook = S3Hook('s3_connection')
    
    source_s3_key_object = hook.get_key(key = f"stop_words/{filename}", bucket_name = 'airflow-sortness')
    file = BytesIO(source_s3_key_object.get()['Body'].read()).read()
    strings = file.decode('utf-8')

    stopwords = strings.split('\n')
    stopwords = [s.strip().replace('\t', ' ') for s in stopwords]

    return stopwords

def get_stopwords():
    stopwords = get_s3_stopwords("korean_stopwords.TXT")
    okt = Okt()

    stopwords_noun = []
    for i in stopwords:
        if okt.nouns(i) is not empty:
            for j in okt.nouns(i):
                stopwords_noun.append(j)
    return stopwords_noun

def tokenizing(**context):
    date = str(context['execution_date'])[:10]
    cursor = conn.cursor()
    sql = "select article_id, content from article where ranking_date = '{0}'".format(date)
    df = pd.read_sql_query(sql, conn)
    
    text_list = []
    df['content'] = df['content'].fillna("")

    for index, row in df.iterrows():
        row['content'] = re.sub('[^가-힣\s\']', ' ', row['content'])
        row['content'] = re.sub('\s{2,}', ' ', row['content'])
        text_list.append(row['content'])
    df["preprocessed_content"] = text_list
    print(text_list) 
    noun_extractor = LRNounExtractor_v2(verbose=True, extract_compound=True)
    nouns = noun_extractor.train(text_list)
    word_score = noun_extractor.extract()
    
    scores = {word: score[1] for word, score in word_score.items()}
    self_stopwords = get_s3_stopwords("self_stopwords.TXT") 

    for word in self_stopwords:
        if word in scores:
            scores[word] = 0
    tokenizer = NounMatchTokenizer(noun_scores = scores)
    word_tokens = []

    for index, row in df.iterrows():
        noun_text = tokenizer.tokenize(row["preprocessed_content"])
        word_tokens.append(noun_text)
    
    stopwords = get_stopwords()
    preprocessed_noun = []
    
    for token in word_tokens:
        doc_token = []
        for word in token:
            if word not in stopwords:
                if len(word) >1:
                    doc_token.append(word)
        preprocessed_noun.append(doc_token)

    okt = Okt()
    tokens = []
    for doc in preprocessed_noun:
        doc_tokens = []
        for word in doc:
            pos_list = okt.pos(word, norm=True, stem = True)
            part_of_speech = [pos[1] for pos in pos_list]
            if 'Noun' in set(part_of_speech) and len(set(part_of_speech)) == 1:
                doc_tokens.append(word)
        tokens.append(str(doc_tokens))
        print(doc_tokens)

    df['final_token2']=tokens

    for index, row in df.iterrows():
        preprocess = re.sub('\[|\]|\'', '', row['final_token2'])
        article_id = row['article_id']
        print(preprocess)
        sql = "insert into token(article_id, voca_list) values(%s, %s)"
        cursor.execute(sql, (article_id, preprocess))
    conn.commit()
    conn.close()

import pandas as pd
from gensim import corpora, models
import pyLDAvis.gensim_models
import pandas as pd

def update_year(**context):
    date = str(context['execution_date'])[:10]
    start_date = date[:4]+"-01-01"
    file_name = date[:4]
    topic_modeling(start_date, date, file_name)

def update_month(**context):
    date = str(context['execution_date'])[:10]
    
    start_date = date[:8]+"01"
    end_date = date
    file_name = date[:8].replace('-', '')
    topic_modeling(start_date, end_date, file_name)

def update_today(**context):
    date = str(context['execution_date'])[:10]

    file_name = "today"
    topic_modeling(date, date, file_name)

def update_quarter(**context):
    date = str(context['execution_date'])[:10]

    if int(date[5:7]) in [1,2,3]:
        start_date = date[:5]+"01-01"
        file_name = date[0:4]+"_1"
    elif int(date[5:7]) in [4,5,6]:
        start_date = date[:5]+"04-01"
        file_name = date[0:4]+"_2"
    elif int(date[5:7]) in [7,8,9]:
        start_date = date[:5]+"07-01"
        file_name = date[0:4]+"_3"
    elif int(date[5:7]) in [10,11,12]:
        start_date = date[:5]+"10-01"
        file_name = date[0:4]+"_4"
    print(start_date, file_name) 
    end_date = date

    topic_modeling(start_date, end_date, file_name)

from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def topic_modeling(start_date, end_date, file_name):
    cursor = conn.cursor() 
    sql = "select article_id from article where ranking_date>='{0}' and ranking_date<='{1}'".format(start_date, end_date)
    cursor.execute(sql)
    result = cursor.fetchall()
    result = [id[0] for id in list(result)]

    sql = "select * from token where article_id in {0}".format(tuple(result))
    cursor.execute(sql)
    result = cursor.fetchall()
    result = [string[1] for string in list(result)]
    print(result)
    tokens = [tokens.split(', ') for tokens in result]
    print(tokens)
    df = pd.DataFrame({'tokenized_docs':tokens})
    dictionary = corpora.Dictionary(df['tokenized_docs'])
    bow_corpus = [dictionary.doc2bow(doc) for doc in df['tokenized_docs']]
    tfidf = models.TfidfModel(bow_corpus, normalize = True)
    corpus_tfidf = tfidf[bow_corpus]
    ldamodel = models.ldamodel.LdaModel(corpus_tfidf, num_topics = 5, id2word = dictionary, passes = 15)
    topics = ldamodel.print_topics(num_words = 4)
    vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus_tfidf, dictionary)
    hook = S3Hook('s3_connection')
    hook.load_string(string_data = pyLDAvis.prepared_data_to_html(vis), key = f'files/{file_name}.html', bucket_name = 'airflow-sortness', replace = True)
dag = DAG(dag_id = "dag_1", default_args=default_args)
t1 = PythonOperator(task_id = 'crawling', python_callable = crawling_main, provide_context = True, dag = dag)
t2 = PythonOperator(task_id = 'tokenizing', python_callable = tokenizing, provide_context = True, dag = dag)
t3 = PythonOperator(task_id = 'update_month_data', python_callable = update_month, provide_context = True, dag = dag)
t4 = PythonOperator(task_id = 'update_today_data', python_callable = update_today, provide_context = True, dag = dag)
t5 = PythonOperator(task_id = 'update_quarter_data', python_callable = update_quarter, provide_context = True, dag = dag)
t6 = PythonOperator(task_id = 'update_year_data', python_callable = update_year, provide_context = True, dag = dag)
t1>>t2>>[t3, t4, t5, t6]
