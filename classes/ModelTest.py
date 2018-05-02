from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
# from sklearn.cluster import  MiniBatchKMeans
# from sklearn.neighbors import KDTree, LSHForest
# from sklearn.cluster import AgglomerativeClustering
import numpy as np
import nltk
import pandas as pd
from nltk import word_tokenize
# from nltk.corpus import stopwords
import re
# import pyLDAvis
from pymongo import MongoClient
# import pyLDAvis.sklearn
from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt
# from sklearn.manifold import MDS
import time
from threading import Thread
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

class Modeling:
    lda_model = None
    tweet_list = []
    tweet_words = []

    def clean(self, data):
        data = re.sub(r'(\n)', '', data)
        data = re.sub(r"[^a-zA-Z'\-]+", ' ', data)
        data = re.sub(r'(R{1}T)', '', data)
        data = re.sub(r'(display text range)', '', data)
        data = data.split("https")[0].strip('\"' + ' ')
        data = ' '.join(word for word in data.split() if word[0] != '@')
        return "".join(data)



    def tokenize_nltk(self, text):
        tokens = word_tokenize(text)
        text = nltk.Text(tokens)
        # stop_words = set(stopwords.words('english'))
        # stop_words.update(custom_stopwords)
        # words = [w.lower() for w in text if w.isalpha() and w.lower() not in stopwords.words('english')]
        # return words
        return text

    def display_topics(self, H, W, feature_names, documents, no_top_words, no_top_documents):
        for topic_idx, topic in enumerate(H):
            print("\nTopic %d:" % (topic_idx))
            # print("\nTop Words:\n")
            # print(" ".join(['\"'+feature_names[i]+'\"' for i in topic.argsort()[:-no_top_words - 1:-1]]))

            print("\nTop Docs: (Doc Id, Doc Text)")
            print()
            tw = []
            tl = []
            tw.append(" ".join(['\"'+feature_names[i]+'\"' for i in topic.argsort()[:-no_top_words - 1:-1]]))
            top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
            for doc_index in top_doc_indices:
                # print(top_doc_indices)
                print(doc_index+2, documents[doc_index])
                tl.append(documents[doc_index])
            Modeling.tweet_list.append(tl)
            Modeling.tweet_words.append(tw)
            print("\n###########################################")


    #
    # def plot_dendrogram(self, model, **kwargs):
    #
    #     # Children of hierarchical clustering
    #     children = model.children_
    #
    #     # Distances between each pair of children
    #     # Since we don't have this information, we can use a uniform one for plotting
    #     distance = np.arange(children.shape[0])
    #
    #     # The number of observations contained in each cluster level
    #     no_of_observations = np.arange(2, children.shape[0] + 2)
    #
    #     # Create linkage matrix and then plot the dendrogram
    #     linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    #
    #     # Plot the corresponding dendrogram
    #     dendrogram(linkage_matrix, **kwargs)


    def run_model(self, documents):
        # no_features = 500


        # documents['body'] = documents['body'].apply(self.clean)
        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        tf_vectorizer = TfidfVectorizer(input='content', analyzer='word', ngram_range=(2, 3),
                                        max_df=.80)
        # tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
        # print(documents.keys)
        tf = tf_vectorizer.fit_transform(documents['text'])
        tf_feature_names = tf_vectorizer.get_feature_names()

        search_params = {'n_components': [3, 5, 10, 15]}
        t1 = time.time()
        # print("time start ", t1)
        lda = LatentDirichletAllocation(max_iter=50, learning_method='online', learning_decay=.7)
        lda_model = GridSearchCV(lda, param_grid=search_params, n_jobs=4)
        lda_model.fit(tf)
        self.lda_model = lda_model.best_estimator_
        t2 = time.time()
        print("time taken: ", t2-t1)
        # Model Parameters
        print("Best Model's Params: ", lda_model.best_params_)

        # Log Likelihood Score
        print("Best Log Likelihood Score: ", lda_model.best_score_)

        # Perplexity
        print("Model Perplexity: ", self.lda_model.perplexity(tf))
        # no_topics = 5
        #
        # # Run LDA
        # Modeling.lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=50, learning_method='online',
        #                                       learning_offset=18, learning_decay=0.6, random_state=0).partial_fit(tf)

        lda_W = self.lda_model.transform(tf)
        lda_H = self.lda_model.components_
        topicnames = ["Topic" + str(i) for i in range(lda_model.best_params_['n_components'])]
        docnames = ["Tweet" + str(i) for i in range(len(documents['id']))]
        df_document_topic = pd.DataFrame(np.round(lda_W, 2), columns=topicnames, index=docnames)

        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        documents['cluster'] = dominant_topic
        #
        # t1 = time.time()
        # svd_model = TruncatedSVD(n_components=2)  # 2 components
        # lda_output_svd = svd_model.fit_transform(lda_W)
        # x = lda_output_svd[:, 0]
        # y = lda_output_svd[:, 1]
        # ac = AgglomerativeClustering(n_clusters=5)
        # model = GridSearchCV(ac, param_grid={'n_clusters': [3, 5, 10, 20, 80]}, n_jobs=5)
        # labels = model.labels_
        # plt.title('Hierarchical Clustering Dendrogram')
        # self.plot_dendrogram(model, labels=labels, truncate_mode='level', p=5)
        # plt.show()
        # print(labels)
        # # print(self.lda_model.__doc__)

        # km = MiniBatchKMeans(n_init=100, max_iter=1000)
        # model = GridSearchCV(km, param_grid={'n_clusters': [3, 5, 10, 20, 80]}, n_jobs=5)
        # model = model.fit(lda_W)
        # km_model = model.best_estimator_

        # # Z = AgglomerativeClustering(3).fit_predict(self.lda_model)
        # # print(Z.dtype)
        # # plt.figure()
        # # dendrogram(Z.astype(float))
        # # plt.show()
        # labels = MiniBatchKMeans(n_clusters=5, random_state=100).fit_predict(lda_W)
        # # labels = clusters.labels_
        # t2 = time.time()
        # print('Time taken: ', t2 - t1)
        # print(set(labels))
        # # # print(documents.shape)
        # documents['cluster'] = labels
        # plt.figure(figsize=(12, 12))
        # plt.scatter(x, y, c=labels)
        # plt.xlabel('Component 2')
        # plt.xlabel('Component 1')
        # plt.title("Segregation of Topic Clusters", )
        # plt.show()
        # print("Top terms per cluster:")
        # km.
        # order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        # terms = tf_feature_names
        # for i in range(5):
        #     print("Cluster %d:" % i,)
        #     for ind in order_centroids[i, :10]:
        #         print(' %s' % terms[ind],)
        #     print()

        # no_top_words = 10
        # no_top_documents = 10
        #
        # self.display_topics(lda_H, lda_W, tf_feature_names, documents['body'], no_top_words, no_top_documents)


#         vis = pyLDAvis.sklearn.prepare(self.lda_model, tf, tf_vectorizer, R=10)
#         thread = Thread(target=threaded_function, args=vis)
#         thread.start()
#         # pyLDAvis.show(vis)
#         return
#
# def threaded_function(arg):
#     pyLDAvis.show(arg)




# if __name__=='__main__':
#     obj = Modeling()
#
#
#     twitter_data2 = pd.read_csv("test.csv", sep=",")
#
#     documents = twitter_data2
#     documents = documents.dropna(subset=['body', 'twid'])
#     # print(documents)
#     obj.run_model(documents)