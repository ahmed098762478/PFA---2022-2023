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

from flask import Flask, render_template, request, jsonify
from flask_mysqldb import MySQL

app = Flask(__name__)
app.static_folder = 'static'

# Configuration de la base de données
app.config['MYSQL_HOST'] = 'localhost' # Adresse de l'hôte MySQL
app.config['MYSQL_USER'] = 'root'   # Nom d'utilisateur MySQL
app.config['MYSQL_PASSWORD'] = ''  # Mot de passe MySQL
app.config['MYSQL_DB'] = 'chatbot'  # Nom de la base de données MySQL

# Initialisation de l'extension MySQL
mysql = MySQL(app)

# Fonctions de gestion de la base de données
def store_reservation(nom, prenom, email, numerotel, jour_reservation, heure, nombre_personnes, nom_restaurant):
    # Connexion à la base de données
    cur = mysql.connection.cursor()

    # Requête SQL pour insérer les informations de réservation dans la table
    sql = "INSERT INTO reservation (nom, prenom, email, numerotel, jour_reservation, heure, nombre_personnes, nom_restaurant) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    values = (nom, prenom, email, numerotel, jour_reservation, heure, nombre_personnes, nom_restaurant)

    # Exécution de la requête SQL
    cur.execute(sql, values)

    # Validation des modifications dans la base de données
    mysql.connection.commit()

    # Fermeture du curseur
    cur.close()


def check_reservation(nom, prenom, email):
    # Connexion à la base de données
    cur = mysql.connection.cursor()

    # Requête SQL pour vérifier si la réservation existe dans la base de données
    sql = "SELECT * FROM reservation WHERE nom = %s AND prenom = %s AND email = %s"
    values = (nom, prenom, email)

    # Exécution de la requête SQL
    cur.execute(sql, values)

    # Récupération des résultats
    result = cur.fetchone()

    # Fermeture du curseur
    cur.close()

    # Vérification si la réservation existe ou non
    if result:
        return True
    else:
        return False



def supprimer_reservation(nom, prenom, email):
    # Connexion à la base de données
    cur = mysql.connection.cursor()

    # Requête SQL pour supprimer la réservation de la base de données
    sql = "DELETE FROM reservation WHERE nom = %s AND prenom = %s AND email = %s"
    values = (nom, prenom, email)

    # Exécution de la requête SQL
    cur.execute(sql, values)

    # Validation des modifications dans la base de données
    mysql.connection.commit()

    # Fermeture du curseur
    cur.close()

    # Vérification si des lignes ont été supprimées
    if cur.rowcount > 0:
        return render_template("annulation_succes.html")
    else:
        return render_template("annulation_error.html")
    

def store_commande(nom, prenom, email, produit, quantite, livraison):
    # Connexion à la base de données
    cur = mysql.connection.cursor()

    # Requête SQL pour insérer les informations de commande dans la table
    sql = "INSERT INTO commande (nom, prenom, email, produit, quantite, livraison) VALUES (%s, %s, %s, %s, %s, %s)"
    values = (nom, prenom, email, produit, quantite, livraison)

    # Exécution de la requête SQL
    cur.execute(sql, values)

    # Validation des modifications dans la base de données
    mysql.connection.commit()

    # Fermeture du curseur
    cur.close()



def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix            
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)     
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
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
        # Récupérer la première réponse du fichier JSON
        res1 = getResponse(ints, intents)
        # Charger le contenu du formulaire de réservation
        res2 = render_template("reservation.html")
        return res1 + '\n\n' + res2
    elif ints[0]['intent'] == 'annulation':
        return redirect("/annuler_reservation")
    elif ints[0]['intent'] == 'choix_plat':
        return redirect("/commande")
    else:
        res = getResponse(ints, intents)
        return res


@app.route("/reservation", methods=['POST'])
def handle_reservation():
    nom = request.form.get("nom")
    prenom = request.form.get("prenom")
    email = request.form.get("email")
    numerotel = request.form.get("numerotel")
    jour_reservation = request.form.get("jour_reservation")
    heure = request.form.get("heure")
    nombre_personnes = request.form.get("nombre_personnes")
    nom_restaurant = request.form.get("nom_restaurant")

    # Appel de la fonction pour stocker les informations de réservation dans la base de données
    store_reservation(nom, prenom, email, numerotel, jour_reservation, heure, nombre_personnes, nom_restaurant)
    
    # Récupération des informations de réservation
    reservation_details = {
        'nom': nom,
        'prenom': prenom,
        'email': email,
        'numerotel': numerotel,
        'jour_reservation': jour_reservation,
        'heure': heure,
        'nombre_personnes': nombre_personnes,
        'nom_restaurant': nom_restaurant
    }
    # Rendu du template reservation_success.html en passant les informations de réservation
    return render_template("reservation_succes.html", reservation=reservation_details)



@app.route("/annuler_reservation", methods=['POST'])
def handle_annuler_reservation():
    nom = request.form.get("nom")
    prenom = request.form.get("prenom")
    email = request.form.get("email")

    # Vérification si les informations de réservation existent dans la base de données
    if check_reservation(nom, prenom, email):
        # Appel de la fonction pour supprimer la réservation dans la base de données
        supprimer_reservation(nom, prenom, email)
        return render_template("annulation_succes.html")
    else:
        return render_template("annulation_error.html")
    

@app.route("/commande", methods=['POST'])
def handle_commande():
    nom = request.form.get("nom")
    prenom = request.form.get("prenom")
    email = request.form.get("email")
    produit = request.form.get("produit")
    quantite = request.form.get("quantite")
    livraison = request.form.get("livraison")

    # Appel de la fonction pour stocker les informations de commande dans la base de données
    store_commande(nom, prenom, email, produit, quantite, livraison)

    # Récupération des informations de commande
    commande_details = {
        'nom': nom,
        'prenom': prenom,
        'email': email,
        'produit': produit,
        'quantite': quantite,
        'livraison': livraison
    }

    # Rendu du template commande_succes.html en passant les informations de commande
    return render_template("commande_succes.html", commande=commande_details)



@app.route("/reservation", methods=['GET'])
def reservation():
    return render_template("reservation.html")


@app.route("/reservation_success", methods=['GET'])
def reservation_success():
    return render_template("reservation_success.html")


@app.route("/annuler_reservation", methods=['GET'])
def annuler_reservation():
    return render_template("annuler_reservation.html")

@app.route("/commande", methods=['GET'])
def humberger():
    return render_template("commande.html")

@app.route("/annulation_success", methods=['GET'])
def annulation_succes():
    return render_template("annulation_succes.html")


@app.route("/annulation_error", methods=['GET'])
def annulation_error():
    return render_template("annulation_error.html")


@app.route("/commande_succes", methods=['GET'])
def commande_success():
    return render_template("commande_succes.html")

# Route pour obtenir la réponse du chatbot
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    response = chatbot_response(userText)

    return response


# Route pour afficher le formulaire de réservation
# Route pour afficher le formulaire de réservation
@app.route("/")
def home():
    return render_template("main.html")


if __name__ == "__main__":
    app.run()