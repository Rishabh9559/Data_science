import nltk
import spacy
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
spacy.cli.download('en_core_web_sm')

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

sentence = 'Natural Language Processing (NLP) is a field that ' \
'combines computer science, artificial intelligence and ' \
'language studies. It helps computers understand, process and ' \
'create human language in a way that makes sense and is useful. ' \
'With the growing amount of text data from social media, ' \
'websites and other sources, NLP is becoming a key tool to gain ' \
'insights and automate tasks like analyzing text or translating ' \
'languages.'


stop_words = set(stopwords.words('english'))
# print(stop_words)
# print(sent_tokenize(sentence))
tokens = word_tokenize(sentence)
# normalize all the words to lowercase
lower_tokens = [t.lower() for t in tokens] 
# filtered_tokens which are not a part of stopwords
filtered_tokens = [t for t in lower_tokens if t.isalpha() and t not in stop_words]
# print(filtered_tokens)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_words = [stemmer.stem(t) for t in filtered_tokens]
lemmatized_words = [lemmatizer.lemmatize(t) for t in filtered_tokens]

# print(stemmed_words)
print(lemmatized_words)


nlp = spacy.load('en_core_web_sm')
doc = nlp(sentence)

for sentence in doc.sents:
    print(sentence)