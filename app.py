import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect

from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

app = Flask(__name__)
app.static_folder = 'static'


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
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


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    if ints[0]['intent'] == 'reservation':
        return redirect("/reservation")
    elif ints[0]['intent'] == 'annulation':
        return redirect("/annuler_reservation")
    else:
        res = getResponse(ints, intents)
        return res 
    

@app.route("/reservation", methods=['GET'])
def reservation():
    return render_template("reservation.html")


@app.route("/reservation_success", methods=['GET'])
def reservation_success():
    return render_template("reservation_success.html")


@app.route("/annuler_reservation", methods=['GET'])
def annuler_reservation():
    return render_template("annuler_reservation.html")


@app.route("/annulation_succes", methods=['GET'])
def annulation_succes():
    return render_template("annulation_succes.html")
 

@app.route("/annulation_error", methods=['GET'])
def annulation_error():
    return render_template("annulation_error.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    response = chatbot_response(userText)

    return response


# Route pour afficher le formulaire de réservation
@app.route("/")
def home():
    return render_template("main.html")


if __name__ == "__main__":
    app.run()
