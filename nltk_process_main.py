# ==============================================
# ===== IMPORTANT INSTALLATION INFORMATION =====
# ==============================================
'''
 1.) For nltk to integrate correctly with Spacy, you MUST install numpy version 1.25.1
     --> !pip install --force-reinstall -v "numpy==1.25.1"
'''

# ==================================
# ===== Necessary Dependencies =====
# ==================================
import nltk, os, math, operator, spacy

# Download the required language packages from nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

#  ----- General formatting (Data setup) -----

# ================================================================
# ===== General Formatting -- Data Settup and Initialization =====
# ================================================================

relative_path_to_file = os.path.join("..", "IU2024-ML-DATATHON/data", "HOPG.txt")
absolute_path_to_file = os.path.realpath(relative_path_to_file)

# Create a variable, text, that holds all the text from a specific file
# IMPORTANT: the 'text' variable is a global variable
with open(absolute_path_to_file, "r+") as file:
    text = file.read()

# =============================
# ===== Package Functions =====
# =============================

def generate_freq_dist(text, rmvSpChar=True):
    """Param [string]: Text to be analyzed and converted into a frequency distriution (Map each word to outcome)
    \n\nParam [bool]: True --> Removes special characters from the frequency distribution
    \n\nOutput [Dictionary]: a loop through output returns character, but indexing the output by loop variable outputs the frequency at which the character occurs"""
    freqDist = nltk.FreqDist(text)
    if rmvSpChar:
        for x in ":,.-[];!'\"\t\n/ ?":
            del freqDist[x]
    return freqDist

def generate_relative_frequency(freqDist):
    """Param [list]: Frequency distribution
    \n\nOutput [list]: Relative frequencies of each respective character found in the frequency distribution"""
    total = float(sum(freqDist.values()))
    relfrq = [ x/total for x in freqDist.values() ]
    return relfrq

def generate_entropy(relfrq, specialCase=False):
    """Param [list]: Relative frequency of a frequency distribution
    \n\nParam [bool]: idk tbh, but changes the outcome and was recommended by Damir himself
    \n\nOutput [float]: Randomness element per character in the frequency distribution"""

    # At the moment, idk what the difference is between the two equations
    if specialCase:
        relfrq = [ 1/len(relfrq) ] * len(relfrq)
    
    return -sum( [ x * math.log(x, 2) for x in relfrq if x > 0 ] )

def generate_pointwise_entropy(relfrq):
    """Param [list]: Relative frequency of a frequency distribution
    \n\nOutput [list]: Pointwise entropy per character in the frequency distribution"""
    return [ -x * math.log(x, 2) for x in relfrq ]

def tokenize(text):
    """Param [string]: Text to be tokenized
    \n\nOutput [list]: Tokenized words in a list"""
    return nltk.word_tokenize(text)

def generate_token_frequency_distribution(tokens, delStop=True):
    """Param [list]: Tokenized list of a given text
    \n\nParam [boolean]: True --> Removes stopwords (Recommended)
    \n\nOutput [tuple]: Frequency distribution of tokenized words"""
    TokenFD = nltk.FreqDist(tokens)

    if delStop:
        stopwords = nltk.corpus.stopwords.words('english')
        for x in stopwords:
            del TokenFD[x]
    return tuple(TokenFD)

def generate_bigrams(tokens, n=2):
    """Param [list]: List of tokens from a given text
    \n\nParam [int]: n-gram classifier (default of 2)
    \n\nOutput [list]: bigram based on the value of n"""
    return list(nltk.ngrams(tokens, n))

def generate_bigram_frequency_distribution(bigrams):
    """Param [list]: Bigram list from tokenized input data
    \n\nOutput [freqDist]: Convert to a list to view data (use ngram_output)"""
    return nltk.FreqDist(bigrams)

def ngram_output(myBigramFD):
    """Outputs in format: (', and', 1271)"""
    return [ (" ".join(ngram), myBigramFD[ngram]) for ngram in myBigramFD ]

def output_bigram_entropy(myBigramFD):
    """Param [list]: List of the bigram frequency distribution
    \n\nOutput [list]: Sorted list of bigram entropy"""
    total = float(sum(myBigramFD.values()))
    exceptions = ["]", "[", "--", ",", ".", "'s", "?", "!", "'", "'ye"]
    results = []
    stopwords = nltk.corpus.stopwords.words('english')
    for x in myBigramFD:
        if x[0] in exceptions or x[1] in exceptions:
            continue
        if x[0] in stopwords or x[1] in stopwords:
            continue
        results.append( (x[0], x[1], myBigramFD[x]/total) )
    sortedresults = sorted(results, key=operator.itemgetter(2), reverse=True)
    return sortedresults

def display_bigram_entropy_results(bigramEntropy, n):
    """Param [list]: Bigram Entropy list of a tokenized bigram list
    \n\nParam [int]: Number of lines to be output (Increasing -> Decreasing)
    \n\nOutput [None]: Just prints the top n results"""
    for x in bigramEntropy[:n]:
        print(x[0], x[1], x[2])

# ===============================================
# ===== Automated Bigram Entropy Generation =====
# ===============================================

def auto_bigram_entropy_generation(text, n=2):
    """Param [string]: Text to be analyzed
    \n\nParam [boolean]: True --> Removes stopwords (Default True)
    \n\nOutput [list]: Bigram Entropy of the analyzed text"""

    tokens = tokenize(text)
    bigrams = generate_bigrams(tokens, n)
    bigramFD = generate_bigram_frequency_distribution(bigrams)
    bigramEntropy = output_bigram_entropy(bigramFD)
    return bigramEntropy

# ================
# ===== Main =====
# ================

display_bigram_entropy_results(auto_bigram_entropy_generation(text), 20)