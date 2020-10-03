"""
This will serve as my notes for cleaning and creating the environments that I need

step 1:
    Create pipenv virtual environment and download all maing packages
        pipenv install numpy pandas spacy tensorflow TextBlob scikit-learn
        pipenv run python3 -m spacy download en_core_web_md
        pipenv run python3 -m textblob.download_corpora 

step 2:
    Data Exploration
        using pandas to read the static or flat file, use read_csv and encoding='latin1'
       
        Set Columns:
            slice the df = df[[0,5]]
            df.columns = ['name1','name2']

        Look at unique value counts for target column:
            df['target_col'].value_counts()
        
        Look for:
            Word Count:
                .apply(lambda x: len(str(x).split()))
            Stop Word Count
                .apply(lambda x: len([word for word in x.split() if word in STOP_WORDS]))
            Char Count
                .apply(lambda x: len(str(x)))
            Punctuation Count
                .apply(lambda x: len([c for c in x.split() if c in string.punctuation]))
            Avrg Word Count
                def (x):
                    words = x.split()
                    word_len = 0
                    for word in words:
                        word_len = word_len + len(word)
                    return word_len/len(words)  
            Most used word
                text = ' '.join(df['Tweets'])
                text = text.split()
                freq_ = pd.Series(text).value_counts()
                
                Most Used:top_20 = freq_[:10]
                Least Used = rare = freq_[freq_.values == 1]

Step 3:
    Data Cleaning
        Turn everyting to lower
        Correct Spelling
        Remove all emails
        Remove all @ first
        Remove all hashtags
        Remove and strip all <foo>
        Remove all HTTPS website protocols -> Further work is needed for this
        Clean and Replace all contrractions
        remove all numerical values
        remove all special characters (punctuations as well)
       
        Remove all super common words
        Remove all rare words
        remove all stop words


        ########################CODE:
        def get_clean_data(x):
            # go through multiple steps of cleaning

            # turn everything into lower case
            x = x.lower()

            # fix all potential spelling mistakes
            x = TextBlob(x).correct()

            # remove all emails
            x = re.sub('([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', "", x)

            # remove all @ first
            x = re.sub(r'@([A-Za-z0-9_]+)', "", x)

            # remove all # first
            x = re.sub(r'#([A-Za-z0-9_]+)', "", x)

            # remove and strip all retweets (RT)
            x = re.sub(r'\brt:\b', '', x).strip()

            # remove all websites
            # TODO: Figure out how it works for all possible website protocols
            x = re.sub(
                r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x)

            # clean and replace with contractions
            x = contractions_replace(x)

            # remove all numerical values
            x = re.sub(r'[0-9]+', "", x)

            # remove all special characters
            x = re.sub(r'[^\w ]+', ' ', x)

            # split aka tokenize our tweets
            x = x.split()

            # We are removed all the workds that are in our top 10
            x = [words for words in x if words not in Top_10]

            # We are rempoving all the words that are not in our rare list
            x = [words for words in x if words not in Least_freq]

            # remove all the words in our STOP_WORDS
            x = [words for words in x if words not in STOP_WORDS]

            return " ".join(x)



"""

#❯ psql -h localhost -p 5432 -U postgres

