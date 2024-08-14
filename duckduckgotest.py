import requests
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

query = "Chester do you know what kind of trees grow in california?"

def extract_keywords(query):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(query)
    keywords = [word for word in word_tokens if word.isalnum()
                and word.lower() not in stop_words]
    return keywords

duckduckgo_result = None
keywords = extract_keywords(query)
print("These are the extracted keywords: " + str(keywords))
search_query = " ".join(keywords)
print("Search query: " + search_query)

try:
    response = requests.get('https://api.duckduckgo.com/', params={
        'q': search_query,
        'format': 'json',
        'no_html': 1,
        'no_redirect': 1,
        'skip_disambig': 1
    })
    print("Response status code:", response.status_code)
    data = response.json()
    print("API Response:", data)
    
    if 'AbstractText' in data and data['AbstractText']:
        duckduckgo_result = data['AbstractText']
    elif 'RelatedTopics' in data and data['RelatedTopics']:
        duckduckgo_result = data['RelatedTopics'][0].get('Text', 'No information found.')
    else:
        duckduckgo_result = 'No relevant information found.'

except Exception as e:
    logging.error(f"Error during DuckDuckGo search: {e}")

print(duckduckgo_result)
