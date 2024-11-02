import nltk, os, math, operator, spacy
nltk.download('punkt_tab')
nltk.download('stopwords')

#  ----- General formatting (Data setup) -----

relative_path_to_file = os.path.join("..", "IU2024-ML-DATATHON/data", "HOPG.txt")
absolute_path_to_file = os.path.realpath(relative_path_to_file)

# Create a variable, text, that holds all the text from a specific file
with open(absolute_path_to_file, "r+") as file:
    text = file.read()


# ----- Helper Functions -----

#Finds Entropy which is the randomness element 
def entropy(p):
    return -sum( [ x * math.log(x, 2) for x in p if x > 0 ] )

def ngram_freq_profile_tuple_python_generator_object(myBigramFD):
    """Outputs in format: (', and', 1271)"""
    return [ (" ".join(ngram), myBigramFD[ngram]) for ngram in myBigramFD ]

def outputBigram(myBigramFD):
    """"Takes a myBigramFD input"""
    for bigram in list(myBigramFD.items())[:20]:
        print(bigram[0], bigram[1])
    print("...")

def outputBigramEntropy(myBigramFD):
    total = float(sum(myBigramFD.values()))
    exceptions = ["]", "[", "--", ",", ".", "'s", "?", "!", "'", "'ye"]
    results = []
    for x in myBigramFD:
        if x[0] in exceptions or x[1] in exceptions:
            continue
        if x[0] in stopwords or x[1] in stopwords:
            continue
        results.append( (x[0], x[1], myBigramFD[x]/total) )
    sortedresults = sorted(results, key=operator.itemgetter(2), reverse=True)
    for x in sortedresults[:20]:
        print(x[0], x[1], x[2])


# ----- Normalization Techniques -----
text = text.lower()

# Generate Frequency Distribution (map each word to outcome)
# freqDist is a dictionary, where x looping through acts as the key
freqDist = nltk.FreqDist(text)

# Deleting Special Characters in FreqDist
for x in ":,.-[];!'\"\t\n/ ?":
    del freqDist[x]

# for x in freqDist:
#     print(x, freqDist[x])

# The resulting number of characters from the frequency distribution
total = float(sum(freqDist.values()))
# print(total)

relfrq = [ x/total for x in freqDist.values() ]
# print(relfrq)


# Compute entorpy of character distribution

# print(entropy([ 1/len(relfrq) ] * len(relfrq)))
# print(entropy(relfrq)) 

# Computer pointwise entorpy for the frequency distribution
entdist = [ -x * math.log(x, 2) for x in relfrq ]
# print(entdist)

# ----- Tokenization Techniques -----
tokens = nltk.word_tokenize(text)

# print(tokens[:20])

# Generate frequency profile on the tokens
myTokenFD = nltk.FreqDist(tokens)
# print(myTokenFD)
print(tuple(myTokenFD.items())[:10])

stopwords = nltk.corpus.stopwords.words('english')
# print(stopwords)

# Remove the stopwords from the freaky distrubtion
for x in stopwords:
    del myTokenFD[x]
# print(tuple(myTokenFD)[:20])

# ----- N-Gram Tokenization -----

myTokenBigrams = nltk.ngrams(tokens, 2)
bigrams = list(myTokenBigrams)
# print(bigrams[:20])

# Same thing as above ^^^
myBigramFD = nltk.FreqDist(bigrams)

# Outputting bigrams
# print(outputBigram(myBigramFD))

# This is a helpful tool (Sorry, Christine, we know the name is great :D)
# print(ngram_freq_profile_tuple_python_generator_object(myBigramFD))

#Output the sorted ngram distribution
# sortedngrams = sorted(ngram_freq_profile_tuple_python_generator_object(myBigramFD), key=operator.itemgetter(1), reverse=True)
# print(sortedngrams[:20])
# print("...")

# Output the ngram entropy distribution categorized by highest to lowest entropy occurence
# outputBigramEntropy(myBigramFD)


# ===== SPACY BEGINNING =====

nlp = spacy.load("en_core_web_sm")
# doc = nlp(u'This is the python generator object!')
# for token in doc:
#     print(token.text, token.lemma_)