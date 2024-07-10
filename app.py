from flask import Flask, render_template, request, jsonify
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import requests


NOMINATIM_URL = 'https://nominatim.openstreetmap.org/reverse'

app = Flask(__name__)
app.static_folder = 'static'

lemmatizer = WordNetLemmatizer()

model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_json, tag):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(intents, ints[0]['intent'])
    if is_location_query(msg):
        location_results = search_location()
        response_text = f"Here are some results for your search: {location_results}"
        return response_text
    else:
        return res

def is_location_query(message):
    location_patterns = [" in ", " near ", " around "]
    return any(pattern in message for pattern in location_patterns)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=['GET', 'POST'])
def get_bot_response():
    user_text = request.args.get('msg')
    response_text = chatbot_response(user_text)
    return jsonify({"response": response_text})


@app.route('/location', methods=['POST'])
def get_location():
    if request.method == 'POST':
        latitude = request.json['latitude']
        longitude = request.json['longitude']

        url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={latitude}&lon={longitude}"
        response = requests.get(url)
        data = response.json()
        address = data['display_name']

        return jsonify({'address': address})


@app.route('/search', methods=['POST'])
def search_location():
    if request.method == 'POST':
        search_query = request.json['query']

        url = f"https://nominatim.openstreetmap.org/search.php?q={search_query}&format=json"
        response = requests.get(url)
        data = response.json()

        return jsonify(data)


if __name__ == "__main__":
    app.run(port=5000)


