import pandas
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import time
from IPython.display import clear_output
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, auc


def read_pickle(profile_pickle):
    return pickle.load(open('cleaned_profiles.p', 'rb'))

def clean_profiles(dataframe, pickleName='cleaned_profiles.p', savePickle=False):
    # removes newlines, breaks, and separates /

    #make list of column names for essay responses
    essays_list=['essay'+repr(i) for i in range(0,10)]

    Nprofiles=len(dataframe['essay0'])
    for i in range(Nprofiles):
        for essaycol in essays_list:
            profile=profiles[essaycol][i]
            if isinstance(profile, str)==True:
                profile=profile.replace("\n", " ")
                profile=profile.replace("<br />", "")
                profile=profile.replace("/", " ")
                profile=profile.replace("about me:", "")
                profile=profile.replace("about you:", "")
                profiles[essaycol][i] = profile
        if i%50==0:
            clear_output()
            print(i,'/',Nprofiles)
    # pickle resulting cleaned profile dataframe
    if savePickle==True:
        picklefile=open(pickleName,'wb')
        pickle.dump(profiles, picklefile)
        picklefile.close()

def labels_to_numeric(profilesDF, colName):
    # column category should be something with 2 - reasonable number of categories, e.g. sex, orientation
    # status, smoking,
    text_ids = profilesDF[colName]
    Nlabels = len(np.unique(text_ids))
    le = preprocessing.LabelEncoder()
    le.fit(text_ids)
    label_id_numeric = le.transform(text_ids)
    return label_id_numeric, text_ids

def training_split(set, frac_training):
    Nprofiles = len(set)
    train_set_size = int(frac_training*Nprofiles)

    train_idx = np.arange(0,train_set_size)
    test_idx = np.arange(train_set_size, Nprofiles)
    train_set = [set[i] for i in train_idx]
    test_set = [set[i] for i in test_idx]

    return train_set, test_set, train_idx, test_idx


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
            TP += 1
    for i in range(len(y_hat)):
        if y_hat[i]==1 and y_actual!=y_hat[i]:
            FP += 1
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==0:
            TN += 1
    for i in range(len(y_hat)):
        if y_hat[i]==0 and y_actual!=y_hat[i]:
            FN += 1
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    return(accuracy, TP, FP, TN, FN)


class prediction:
    def __init__(self, profiles, colLabel, frac_training=0.8):
        self.numeric_ids, self.text_ids = labels_to_numeric(profiles, colLabel)
        self.train_prediction, self.test_prediction, self.train_prediction_idx, self.test_prediction_idx = \
            training_split(self.numeric_ids, frac_training)

        self.id_pairs = list( set( list((zip(self.numeric_ids, self.text_ids)) ) ) )


class essay:
    '''
    initialize with
        profiles: the object loaded from the profile pickle
        essay_num: number of the essay response (0-9)
    '''

    def __init__(self, profiles, essay_num, frac_training=0.8):
        self.essay_num = essay_num
        essay_str='essay'+str(essay_num)
        if isinstance(essay_num, int )==False or essay_num>9:
            raise('Need essay number to be an integer from 0-9')
        # stop words
        self.stopWords = set(stopwords.words('english'))
        # punctuation
        self.punctuation = ['?','!','`',':', ',' ,'.','=','+',';' ,'(' ,')', '&', \
                            '$' , '...', "n't", "'s", "'m", "'ll"]
        # column of dataframe containing indivduals responses to essay N
        self.corpus = profiles[essay_str]

        # split into training and test sets
        # initializes self.train_corpus, self.test_corpus, self.train_idx, self.test_idx
        self.train_test_split(frac_training)

        # initialize empty lists for prediction categories and their models
        self.predictions_labels, self.prediction_models, self.prediction_yScores = [], [], []
        self.prediction_vectorizers, self.prediction_tdfif = [] , []

    def add_prediction(self, label, model, vectorizer, train_tfidf, test_tfidf):
        # once you have a model for predicting classification for some column, add the name of the column to the
        # 'predictions_labels' list and add the model to the 'predictions_models'

        # if that prediction already exists, overwrite the model at the appropriate list index
        if label in self.predictions_labels:
            match_idx = self.predictions_labels.index(label)
            self.prediction_models[match_idx] = model
            self.prediction_yScores[match_idx] = model.predict_proba( test_tfidf )
            self.prediction_vectorizers[match_idx] = vectorizer
            self.prediction_tdfif[match_idx] = (train_tfidf, test_tfidf)

        # if not already in lists, append label and model
        else:
            self.predictions_labels.append(label)
            self.prediction_models.append(model)
            self.prediction_yScores.append(model.predict_proba(test_tfidf ))
            self.prediction_vectorizers.append(vectorizer)
            self.prediction_tdfif.append( (train_tfidf, test_tfidf) )

    def retrieve_vectorizer(self, label):
        if label in self.predictions_labels:
            match_idx = self.predictions_labels.index(label)
            return self.prediction_vectorizers[match_idx]
        else:
            print('Vectorizer not saved for ',label,' prediction.')

    def retrieve_tfidf(self, label):
        # returns (training tfidf, test_tfidf)
        if label in self.predictions_labels:
            match_idx = self.predictions_labels.index(label)
            return self.prediction_tdfif[match_idx]
        else:
            print('Weights not saved for ',label,' prediction.')

    def retrieve_model(self, label):
        # if model is saved for column label already, get back the saved model from self.prediction_models
        if label in self.predictions_labels:
            match_idx = self.predictions_labels.index(label)
            return self.prediction_models[match_idx]
        else:
            print('Model not saved for ',label,' prediction.')

    def retrieve_yScore(self, label):
        # if model is saved for column label already, get back the saved model from self.prediction_models
        if label in self.predictions_labels:
            match_idx = self.predictions_labels.index(label)
            return self.prediction_yScores[match_idx]
        else:
            print('Scores not saved for ',label,' prediction.')

    def stem_and_remove_punctuation(self, stemmer=SnowballStemmer("english"), showTime=True  ):
        if showTime==True:
            start_time = time.time()
        # function to stem and remove punctuation and contractions from a single response
        def stem_rm_punc(response, stemmer):
            if isinstance(response, str)==True:
                words = word_tokenize(response)
                wordsFiltered = []

                # since I have a gut feeling that certain types of people may use '?' or '!' more when writing,
                # I'll replace those with strings that hopefully will get picked up as features
                for w in words:
                    if w=='!':
                        wordsFiltered.append('exclamationpoint')
                    elif w=='?':
                        wordsFiltered.append('questionmark')
                    elif w not in self.stopWords and w not in self.punctuation:
                        stemmed_word = stemmer.stem(w)   # stem words (e.g. eating --> eat)
                        wordsFiltered.append(stemmed_word)
                spacing = " "
                return spacing.join(wordsFiltered)

            # else if the response is left blank (NaN), return stop word 'i' just so we don't get
            # stuck with a 'not string' error later on
            else:
                return('i')

        temp_corpus = [stem_rm_punc(response, stemmer) for response in self.corpus]
        self.corpus = temp_corpus
        del temp_corpus

        print('Removed punctuation, tokenized, and stemmed responses for essay ', str(self.essay_num) )
        if showTime==True:
            print('Time Elapsed: {0:.2f}s'.format(time.time()-start_time))

    def train_test_split(self, frac_training):
        self.train_corpus, self.test_corpus, self.train_idx, self.test_idx = \
            training_split(self.corpus, frac_training)


    def vectorizer_init(self, ngram_max=2, min_df=0.01, max_df=0.9):
        # parameters for vectorizer
        ANALYZER = "word" # unit of features are single words rather then phrases of words
        STRIP_ACCENTS = 'unicode'
        TOKENIZER = None
        NGRAM_RANGE = (0,ngram_max) # default 0-2, Range for n-grams
        MIN_DF = min_df # default 0.01, Exclude words that are contained in less that x percent of documents
        MAX_DF = max_df  # default 0.9, Exclude words that are contained in more than x percent of documents

        vectorizer = CountVectorizer(analyzer=ANALYZER,
                            tokenizer=None, # already did this
                            ngram_range=NGRAM_RANGE,
                            stop_words = stopwords.words('english'),  # removed these already except for NaN standins
                            strip_accents=STRIP_ACCENTS,
                            min_df = MIN_DF,
                            max_df = MAX_DF)
        return vectorizer

    def transformer_init(self):
        # using term frequency - inverse document frequency (TFIDF) to weight word features
        NORM = None #turn on normalization flag
        SMOOTH_IDF = True #prvents division by zero errors
        SUBLINEAR_IDF = True #replace TF with 1 + log(TF)
        USE_IDF = True #flag to control whether to use TFIDF  -- weighting by discrimination power of words
        self.transformer = TfidfTransformer(norm = NORM,smooth_idf = SMOOTH_IDF,sublinear_tf = True)


    def bagWords_and_transform(self, vectorizer, showTime=True):
        # get the bag-of-words from the vectorizer and
        # then use TFIDF to weight the tokens found throughout the text
        if showTime==True:
            start_time = time.time()
        train_bag_of_words = vectorizer.fit_transform( self.train_corpus  )
        test_bag_of_words = vectorizer.transform( self.test_corpus )

        train_tfidf = self.transformer.fit_transform(train_bag_of_words)
        test_tfidf = self.transformer.transform(test_bag_of_words)

        return train_tfidf, test_tfidf


    def logRegression(self, prediction, train_tfidf, test_tfidf, reg_penalty='l2'):
        # pass a prediction object to this function
        # training prediction is the training set of the numerical labels for the thing you
        # are trying to predict, e.g. gender column split into training and test sets

        # fit a regularized logistic regression model (default is L2 penalty)
        clf = LogisticRegression(penalty=reg_penalty)
        #fit the model to the training data
        mdl = clf.fit(train_tfidf, prediction.train_prediction)
        yScore = mdl.predict_proba( test_tfidf )

        return mdl, yScore

    def plotROC(self, label, prediction):

        # ROC curve plotting function from NLP workshop
        def plot_roc(y_true, y_score):
            """
            Plots the precision and recall as a function
            of the percent of data for which we calculate
            precision and recall
            """
            # Compute micro-average ROC curve and ROC area
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = roc_auc_score( y_true, y_score)


            plt.figure(figsize=(4,4))
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.01])
            #plt.xlabel(r'False Positive Rate')
            #plt.ylabel(r'True Positive Rate')
            auc_str= ("%.2f" % roc_auc)
            plt.title( 'AUC = ' + auc_str)
            #plt.legend(loc="lower right")
            plt.show()

        y_true = prediction.test_prediction
        try:
            yScore = self.retrieve_yScore(label)
            plot_roc(y_true, yScore[:,1])
        except:
            print('Y scores not save for ',label,'. Try execute add_prediction(label, model) first?' )



    def fit_with_bestMinDF(self, prediction, label, ngram_max, minDF_range, maxDF=0.9, showTime=True):
        if showTime==True:
            start_time = time.time()

        # loop through values of minDF in range and save the one with the best auc
        eval_measures = []
        y_true = prediction.test_prediction

        #try:
        for minDF in minDF_range:
            vectorizer = self.vectorizer_init(ngram_max, min_df=minDF, max_df=maxDF)
            self.transformer_init()
            train_tfidf, test_tfidf = self.bagWords_and_transform(vectorizer)
            mdl,yScore = self.logRegression(prediction, train_tfidf, test_tfidf)
            # if binary classifier, use roc area under curve to evaluate model
            if len(np.unique(prediction.numeric_ids==2)):
                eval_measures.append(roc_auc_score(y_true, yScore[:,1]))
            # if >2 classes of prediction, use correct/total = accuracy measure
            else:
                accuracy = perf_measure(y_true, yScore[:,1])[0]
                eval_measures.append(  )

            #yScore = self.retrieve_yScore(label)

        index_bestAUC = np.argmax(eval_measures)
        self.best_minDF = minDF_range[index_bestAUC]
        vectorizer = self.vectorizer_init(ngram_max, min_df=self.best_minDF, max_df=0.9)
        #transformer_init()
        train_tfidf, test_tfidf = self.bagWords_and_transform(vectorizer)
        mdl,yScore = self.logRegression(prediction, train_tfidf, test_tfidf)
        self.add_prediction(label, mdl, vectorizer, train_tfidf, test_tfidf)

        print('Best min_df: ', self.best_minDF)

        #except:
        #    print('Error... Try execute add_prediction(label, model) first?' )
        if showTime==True:
            print('Time Elapsed: {0:.2f}s'.format(time.time()-start_time))

    def get_top_features(self, label, prediction, Ntop_features):
        # print out the top Ntop_features for a prediction category
        vectorizer = self.retrieve_vectorizer(label)
        features_stem = vectorizer.get_feature_names()
        mdl = self.retrieve_model(label)
        #zipped_coef_features = list( zip(mdl.coef_[0], features_stem),  )
        #zipped_coef_features.sort()
        print('Top feature words for category: ', label)

        if mdl.coef_.shape[0]>2:
            for i, id_pair in enumerate(prediction.id_pairs):
                print(i, id_pair)
                id_num, id_text = id_pair[0], id_pair[1]

                zipped_coef_features = list( zip(mdl.coef_[i], features_stem),  )
                zipped_coef_features.sort()

                print('Top ',repr(Ntop_features),' feature words for ',id_text,' profiles:')
                for f in range(0,Ntop_features):
                    print(zipped_coef_features[f][1], ',  coef=',zipped_coef_features[f][0])
                print()
        # only one coefficient is saved for binary classification
        else:
                id_pairs = prediction.id_pairs
                id_text0, id_text1 = id_pairs[0][1], id_pairs[1][1]

                zipped_coef_features = list( zip(mdl.coef_[0], features_stem),  )
                zipped_coef_features.sort()

                print('Top ',repr(Ntop_features),' feature words for ',id_text0,' profiles:')
                for f in range(0,Ntop_features):
                    print(zipped_coef_features[f][1], ',  coef=',zipped_coef_features[f][0])
                print()

                print('Top ',repr(Ntop_features),' feature words for ',id_text1,' profiles:')
                for f in range(0,Ntop_features):
                    print(zipped_coef_features[len(zipped_coef_features)-f-1][1],',  coef=',zipped_coef_features[len(zipped_coef_features)-f-1][0] )
                print()
