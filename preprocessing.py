import re
import string
from itertools import groupby

import nltk
import pandas as pd

# import ssl
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download()

english_vocab = set(w.lower() for w in nltk.corpus.words.words())


class PreProcessing:

    def preprocess(self, excel_file_path, is_training_data, sheet_index, use_cols, column_names, skip_rows, keep_labels, id_exists):

        df = pd.read_excel(excel_file_path, header=None, usecols=use_cols, sheet_name=sheet_index, skiprows=skip_rows, names=column_names)

        # drop empty rows
        df = df.dropna()

        if id_exists:
            id_col = column_names[0]

        # last column should always be the class label name
        label_col_name = column_names[-1]

        # remove all the rows that are not in the keep labels
        if not is_training_data:
            df = df[df[label_col_name].isin(keep_labels)]

        # row number
        row_num = 0

        # regex = re.compile(r'<.*?>|https?[^ ]+|([@])[^ ]+|[^a-zA-Z#\' ]+|\d+/?')

        tweet = []
        class_label = []
        t_id = []

        negative = 0
        positive = 0
        neutral = 0
        invalid = 0

        for index, row in df.iterrows():
            txt_tweet = re.sub(',', ' ', row['Tweet'])

            processed_tweet = self._pre_processing(txt_tweet)

            if not is_training_data:
                label = row[label_col_name]
                class_label.append(label)

                if int(row[label_col_name]) == 0:
                    neutral += 1
                elif int(row[label_col_name]) == 1:
                    positive += 1
                elif int(row[label_col_name]) == -1:
                    negative += 1
                else:
                    invalid += 1

            row_num += 1
            tweet.append(processed_tweet)
            if id_exists:
                t_id.append(row[id_col])
            else:
                t_id.append(index)

        print(f"Class distribution. Positive {positive} Negative {negative} Neutral {neutral} Invalid {invalid}")

        print("total rows processed %s " % row_num)
        return [tweet, class_label], t_id

    def clean_stop_words(self, tweet):
        self._stop_words = set()

        # self._stop_words.update(nltk.corpus.stopwords.words('english'))
        self._stop_words.add('rt')
        self._stop_words.add('retweet')
        self._stop_words.add('e')
        self._stop_words.add('obama')
        self._stop_words.add('romney')

        return [word for word in tweet if word.lower() not in self._stop_words]

    def remove_links(self, tweet):
        tweet = re.sub(r'http\S+', ' ', tweet)
        return tweet

    def expanding_acronyms(self, curr_tweet):
        curr_tweet = re.sub("w/o", "without ", curr_tweet)
        curr_tweet = re.sub("w/", "with ", curr_tweet)

        return curr_tweet

    def _pre_processing(self, tweet):

        # remove comma
        tweet = re.sub(',', ' ', tweet)

        # remove hashtags
        tweet = re.sub(r'#\S+', ' ', tweet)
        # expanding acronyms
        tweet = self.expanding_acronyms(tweet)
        # tweet = self.remove_links(tweet)

        # removing html tags
        tweet = re.sub(r'<.*?>', ' ', tweet)

        # removing links
        tweet = re.sub(r'http\S+', '', tweet)

        # Removing mentions
        tweet = re.sub("@[A-Za-z0-9_]+", "", tweet)

        # Removing hashtags
        tweet = re.sub("#[A-Za-z0-9_]+", "", tweet)

        # Removing numbers
        tweet = re.sub("\d+/?", "", tweet)

        tweet = re.sub('[%s]' % re.escape(string.punctuation), ' ', tweet)
        tweet = ''.join(''.join(s)[:2] for _, s in groupby(tweet))
        tweet = tweet.encode('ascii', errors='ignore').decode('utf-8').lower()

        t_list = tweet.split()

        # TODO when we are not removing the stop words I am getting a 59% of accuracy
        # t_list = self.clean_stop_words(t_list)

        ps = nltk.stem.PorterStemmer()
        t_list = [ps.stem(word) for word in t_list]

        # wn = nltk.WordNetLemmatizer()
        # t_list = [wn.lemmatize(word) for word in t_list]

        # t_list = self.clean_stop_words(t_list)

        cleaned_tweet = ' '.join(t_list)

        # curr_tweet = self.porter_stemmer(curr_tweet)
        return cleaned_tweet
