import json
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

app = Flask(__name__)

books_neg = 'C:/Users/Bikram/Desktop/NLP_project/Assessment3_data/books/negative.txt'
books_pos = 'C:/Users/Bikram/Desktop/NLP_project/Assessment3_data/books/positive.txt'
books_unl = 'C:/Users/Bikram/Desktop/NLP_project/Assessment3_data/books/unlabeled.txt'
dvd_neg = 'C:/Users/Bikram/Desktop/NLP_project/Assessment3_data/dvd/negative.txt'
dvd_pos = 'C:/Users/Bikram/Desktop/NLP_project/Assessment3_data/dvd/positive.txt'
dvd_unl = 'C:/Users/Bikram/Desktop/NLP_project/Assessment3_data/dvd/unlabeled.txt'
electronics_neg = 'C:/Users/Bikram/Desktop/NLP_project/Assessment3_data/electronics/negative.txt'
electronics_pos = 'C:/Users/Bikram/Desktop/NLP_project/Assessment3_data/electronics/positive.txt'
electronics_unl = 'C:/Users/Bikram/Desktop/NLP_project/Assessment3_data/electronics/unlabeled.txt'
kitchen_pos = 'C:/Users/Bikram/Desktop/NLP_project/Assessment3_data/kitchen_&_housewares/negative.txt'
kitchen_neg = 'C:/Users/Bikram/Desktop/NLP_project/Assessment3_data/kitchen_&_housewares/positive.txt'
kitchen_unl = 'C:/Users/Bikram/Desktop/NLP_project/Assessment3_data/kitchen_&_housewares/unlabeled.txt'

def get_data(file, label=None):
  """ Load data from txt tp json using  fn.convert_to_json, then create dataframe with colums 'Text' and 'Label'"""
  df = pd.DataFrame(data_collect(convert_to_json(file), label), columns=['Text', 'Label'])
  df['Text'] = df['Text'].drop_duplicates()
  df = df.dropna(subset=['Text'])

  return df

def convert_to_json(file):
  """ The func to convert txt to json
  """
  with open(file, 'r') as json_file:
    data = json.load(json_file)
  return data

def data_collect(file, label):
  """ The func to collect data from json file to list with tupls in format of ({text}, label)
  """
  result = []
  for item in file["review"]:
    if "review_text" in item and len(item['review_text'])>0:
      text = item['review_text']
      result.append((text, label))
  return result

def preprocess_text(text: str) -> str:
  """1/ Tokenize the text 2/ Remove stop words 3/ Lemmatize the tokens 4/ Join the tokens back into a string"""
  tokens = word_tokenize(text.lower())
  filtered_tokens = [token for token in tokens if token not in stopwords.words('english') and token.isalnum()]   #nltk.corpus.stopwords.words("english")
  lemmatizer = WordNetLemmatizer()
  lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
  processed_text = ' '.join(lemmatized_tokens)
  return processed_text

def extract_features(text):
    words = word_tokenize(text)
    # words = [word for word in words if word.isalnum() and word not in stopwords.words('english')]
    features = {}
    for word in words:
        features[word] = True
    return features

def check_text(text: str) -> int:
  new_text_transformed = vectorizer.transform([preprocess_text(text)])
  new_text_predicted = classifier.predict(new_text_transformed)
  if new_text_predicted[0] > 0:
    return int(1)
  return int(0)

bookspd_neg = get_data(books_neg, 0)
bookspd_pos = get_data(books_pos, 1)
bookspd_unl = get_data(books_unl)
dvdpd_neg = get_data(dvd_neg, 0)
dvdpd_pos = get_data(dvd_pos, 1)
dvdpd_unl = get_data(dvd_unl)
electronicspd_neg = get_data(electronics_neg, 0)
electronicspd_pos = get_data(electronics_pos, 1)
electronicspd_unl = get_data(electronics_unl)
kitchenpd_neg = get_data(kitchen_neg, 0)
kitchenpd_pos = get_data(kitchen_pos, 1)
kitchenpd_unl = get_data(kitchen_unl)
df = pd.concat([bookspd_neg,
                bookspd_pos,
                dvdpd_neg,
                dvdpd_pos,
                electronicspd_neg,
                electronicspd_pos,
                kitchenpd_neg,
                kitchenpd_pos],
                ignore_index=True)
df['Text'] = df['Text'].apply(preprocess_text)

df_unl = pd.concat([bookspd_unl, dvdpd_unl, kitchenpd_unl], ignore_index=True)
df_unl['Text'] = df_unl['Text'].apply(preprocess_text)
df_unl['Text'] = df_unl['Text'].str.replace('``', '')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

X = df['Text']
y = df['Label']
vectorizer = CountVectorizer()  # You can also use TfidfVectorizer()
X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156)
classifier = MLPClassifier(max_iter=400)  #MultinomialNB() MLPClassifier(max_iter=400)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(F"{accuracy:.2%}")



df_unl['Label'] = df_unl['Text'].apply(check_text)
#df_unl = df_unl[df_unl['Text'].apply(lambda x: len(x.split()) > 10)] # We can clean data from sent < 7 words

df_new = pd.concat([df, df_unl], ignore_index=True)

X = df_new['Text']
y = df_new['Label']
vectorizer = CountVectorizer()  # You can also use TfidfVectorizer()
X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156)
classifier = MLPClassifier(max_iter=400)  #MultinomialNB() MLPClassifier(max_iter=400)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(F"{accuracy:.2%}")

print('Prepeared pos - ', check_text('Avengers: Endgame is not just a culmination of the last eleven years of the Marvel Studios cinematic saga but also a celebration of everything people have come to love about these characters.'))
print('Prepeared pos - ',check_text("The conclusion of Infinity War is so shocking because it doesn't feel like a cliffhanger, more like a drastic wiping clean of the slate before the whole cycle starts again, with whatever reversal of fortune or comprehensive reboot it may be."))
print('Prepeared pos short- ', check_text('I love this world very much'))
print('Prepeared neg with sarcasm- ', check_text("I noticed that you didn’t follow the established process for this task and it didnt come out as expected. Do you want to review the correct process together to make sure it comes out better next time?"))
print('Prepeared neg - ', check_text('this is a vary bad wovie!'))

#filename1 = 'mlp_classifier_model.joblib'
#filename2 = 'mlp_vectorizer_model.joblib'
##joblib.dump(classifier, filename1)
#joblib.dump(vectorizer, filename2)


#classifier = joblib.load(filename1)
#vectorizer = joblib.load(filename2)

@app.route('/', methods=["GET", "POST"]) # Specify allowed methods
def main():
    if request.method == "POST":
        inp = request.form.get("inp")
        preprocessed_text = preprocess_text(inp)
        new_text_transformed = vectorizer.transform([preprocessed_text])
        new_text_predicted = classifier.predict(new_text_transformed)
        if new_text_predicted[0] > 0:
            return render_template('home.html', message="Positive😄")
        else:
            return render_template('home.html', message="Negative😣")
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)