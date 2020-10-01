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
            least used word
"""
