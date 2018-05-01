from gevent import monkey; monkey.patch_all()
import queue 
import logging
import sys
import os
import time 
import threading 
# import Stemmer
from pymongo import MongoClient
from sklearn.linear_model import SGDClassifier
from gensim import corpora
from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import pandas as pd

# sys.path.append('active_stream/')
# os.chdir('/Users/Shehroz/Desktop/active_stream-master/active_stream')
from classes.streaming import Streamer, Listener
from classes.annotation import Annotator
from classes.credentials import credentials
from classes.text_processing import TextProcessor
from classes.monitor import Monitor
from classes.classification import Classifier, Trainer
from classes.ModelTest import Modeling


async_mode = None
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='gevent', logger=False)
thread = None



@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', async_mode=socketio.async_mode)

@socketio.on('connect')
def connected():
    logging.info('Received connect request')
    emit('log', {'data': 'Connected'})

@socketio.on('tweet_relevant')
def tweet_relevant():
    logging.debug('Received: tweet_relevant')
    emit('log', {'data': 'Connected'})
    data['queues']['annotation_response'].put('relevant')

@socketio.on('tweet_irrelevant')
def tweet_irrelevant():
    logging.debug('Received: tweet_irrelevant')
    data['queues']['annotation_response'].put('irrelevant')

@socketio.on('refresh')
def tweet_irrelevant():
    logging.debug('Received refresh')
    data['queues']['annotation_response'].put('refresh')

@socketio.on('skip')
def tweet_irrelevant():
    logging.debug('Received skip')
    data['queues']['annotation_response'].put('skip')

@socketio.on('connect')
def test_connect():
    global annotator
    if not annotator:
        time.sleep(10)
    if annotator.is_alive():
        # annotator.resume()
        logging.debug('Annotator already alive. Refreshing')
        emit('keywords', {'keywords': list(streamer.keywords)})
        annotator.first = True
    else:
        logging.info('Starting Annotator.')
        emit('keywords', {'keywords': list(streamer.keywords)})
        annotator.start()

@socketio.on('disconnect')
def test_disconnect():
    logging.info('disconnecting'+ '*' * 10)
    global annotator
    logging.info('Pausing annotator')
    annotator.pause()

@socketio.on('add_keyword')
def add_keyword(message):
    logging.debug('Received request to add new keyword. Sending to Streamer.')
    data['queues']['keywords'].put({'add': True, 'word': message['data']})

@socketio.on('remove_keyword')
def remove_keyword(message):
    logging.debug('Received request to remove keyword. Sending to Streamer.')
    data['queues']['keywords'].put({'add': False, 'word': message['data']})
#
# @app.route('/kuchbh_button', methods=['GET', 'POST'])
# def resultgetter():
#     client = MongoClient('localhost', 27017)
#     db = client.HAKET_stream
#     # db = client.FYP
#     collection = db.data
#     r_tweets = []
#     x = collection.find({'classifier_relevant' : True})
#     # text  = []
#     #
#     # for i in x:
#     #     # a = i['id'] + " : " + i['text']
#     #     tweets.append([i['user'],i['text'],i['classifier_relevant'],i['manual_relevant']])
#     #     logging.debug(i['user'])
#     # logging.debug(tweets)
#
#     return render_template('result.html',tweets_array=x)

@socketio.on('result_show')
def Results():
    # global streamer
    # streamer.pause()
    global annotator
    annotator.join

    client = MongoClient('localhost', 27017)
    db = client.HAKET_stream
    collection = db.data
    print('************////////////////////////////***************')
    # mongobatchupdate()

    x = collection.find({'classifier_relevant': True})
    df = pd.DataFrame(list(x))
    # logging.info(df.head())
    print('////////////////////////////***************////////////////////////////////')
    col_list = ['id', 'text']
    # col_list2 = ['username', 'text']
    df = df[col_list]
    # print(df)
    obj = Modeling()

    # df = df.dropna(subset=['body', 'twid'])
    obj.run_model(df)
    print('////////////////////////////********printing*******////////////////////////////////')
    clus = set(list(df.cluster))
    print(list(clus))
    clus = list(clus)

    realclusters = []
    for i in clus:
        cluster = df[df.cluster ==i]
        # print(i)
        realclusters.append(cluster)


    print(realclusters)

    #return render_template("result.html", clusters=realclusters)
    #return render_template("result.html", clusters=realclusters)
    #emit('redirect', {'url': url_for('result.html')})



@app.before_first_request
def main():

    BUF_SIZE = 1000
    db = 'HAKET_stream'
    collection = 'data'
    filters = {'languages': ['en'], 'locations': []}
    n_before_train = 1


    data = {
            'database': MongoClient()[db][collection],
            'queues': {
                'text_processing': queue.Queue(BUF_SIZE),
                'model': queue.Queue(1),
                'annotation_response': queue.Queue(1),
                'most_important_features': queue.Queue(1),
                'keywords': queue.Queue(BUF_SIZE),
                'limit': queue.Queue(BUF_SIZE),
                'messages': queue.Queue(BUF_SIZE)
                },
            'dictionary': corpora.Dictionary(),
            'events': {
                'train_model': threading.Event()
                },
            'filters': filters,
            'socket': socketio,
            }


    data['database'].drop()


    logging.basicConfig(level=logging.DEBUG,
                     format= ''#'%(asctime)s (%(threadName)s) %(message)s',
                    # filename='debug.log'
                    )


    logging.info('\n'*5)
    logging.info('*'*10 + 'ACTIVE LEARNING' + '*'*10)
    logging.info('Starting Application...')


    # Initialize Threads

    streamer = Streamer(credentials=credentials['coll_1'], data=data)

    text_processor = TextProcessor(data)
    annotator = Annotator(train_threshold=n_before_train, data=data)
    classifier = Classifier(data)
    monitor = Monitor(streamer=streamer, classifier=classifier,
                      annotator=annotator, data=data)
    trainer = Trainer(data=data, streamer=streamer,
                      clf=SGDClassifier(loss='log', penalty='elasticnet'))

    threads = [streamer, text_processor, monitor, classifier, trainer]
    check = True

    for t in threads:
        logging.info('Starting {t.name}...')
        logging.info('*' * 10 + 'THREAD STARTING' + '*' * 10)
        if (t.isAlive() == False):
            t.start()
        else:
            t.resume()
    # startproject(threads, app)
    try:
        # logging.info('Starting interface...')
        socketio.run(app, debug=False, log_output=False)
    except KeyboardInterrupt:
        # logging.info('Keyboard Interrupt. Sending stoprequest to all threads')
        annotator.join()
        for t in threads:
            logging.debug('Sending stoprequest to ',{t.name})
            t.join()
        logging.info('Done')
        sys.exit('Main thread stopped by user.')


if __name__ == '__main__':
    main()