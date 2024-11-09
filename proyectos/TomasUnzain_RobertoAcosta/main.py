import re
import pandas as pd
import numpy as np
import time
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os

# Crear la carpeta 'Resultados' si no existe
resultados_dir = 'Resultados'
os.makedirs(resultados_dir, exist_ok=True)

# Descargar el lexicón de VADER si aún no está disponible
nltk.download('vader_lexicon')

# Diccionarios de abreviaturas y contracciones
abbreviation_dict = {
    "idk": "I do not know",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "fyi": "for your information",
    "omg": "oh my god",
    "lol": "laughing out loud",
    "btw": "by the way",
    "brb": "be right back",
    "lmao": "laughing my ass off",
    "nvm": "never mind",
    "tbh": "to be honest",
    "smh": "shaking my head",
    "dm": "direct message",
    "afaik": "as far as I know",
    "ikr": "I know right",
    "wtf": "what the fuck",
    "rt": "",
    "wysiwyg": "what you see is what you get",
    "texn": "technology",
    "lt": "less than",
    "rds": "relational database system",
    "hmu": "hit me up",
    "bff": "best friends forever",
    "ftw": "for the win",
    "irl": "in real life",
    "jk": "just kidding",
    "np": "no problem",
    "rofl": "rolling on the floor laughing",
    "tba": "to be announced",
    "tbd": "to be determined",
    "afk": "away from keyboard",
    "bbl": "be back later",
    "bfn": "bye for now",
    "omw": "on my way",
    "thx": "thanks",
    "ttyl": "talk to you later",
    "gg": "good game",
    "g2g": "got to go",
    "atm": "at the moment",
    "gr8": "great",
    "b4": "before",
    "ur": "your",
    "u": "you",
    "cya": "see you",
    "txt": "text",
    "plz": "please",
    "cu": "see you",
    "bday": "birthday",
    # New entries
    "dx": "Deluxe",             # Used in product model names, e.g., Kindle DX
    "espn": "Entertainment and Sports Programming Network",  # Sports network
    "pyt": "Pretty Young Thing",  # Song title (Michael Jackson)
    "jus": "just",               # Slang
    "ttiv": "", 
    "fav": "favorite",
    "omgg": "oh my god",
    "zomg": "oh my god",
    "io": "Input/Output",
    "lmao": "laughing my ass off",
    "lol": "laughing out loud",
    "rd": "third",
    "wtf": "what the heck",
    "vios": "Verizon Fios",
    "dvr": "Digital Video Recorder",
    "gm": "General Motors",
    "pm": "afternoon/evening",
    "mtv": "Music Television",
    "txt": "text message",
    "dr": "doctor",
    "f__k": "heck",
    "fml": "forget my life",
    "ugh": "expression of frustration",
    "hr": "hour",
    "smh": "shaking my head",
    "nba": "National Basketball Association",
    "bt": "but",
    "gt": "greater than",
    "monsta": "monster",
    "d": "Digital",
    "btw": "by the way",
    "u": "you",
    "cuz": "because",
    "smh": "shaking my head",
    "summize": "search feature",
    "lawnmowing": "cutting grass",
    "sooo": "so",
    "hr": "hour",
    "blech": "expression of disgust",
    "f up": "mess up",
    "fucking": "hecking",
    "ughh": "expression of frustration",
    "comcast": "internet provider",
    "arg": "expression of frustration",
    "quaint": "old-fashioned",
    "nin": "Nine Inch Nails",
    "arhh": "expression of frustration",
    "atebits": "software developer",
    "np": "no problem",
    "dude": "guy",
    "oooooooh": "oh",
    "friggin": "freaking",
    "wat": "what",
    "grr": "expression of frustration",
    "katydids": "type of insect",
    "barraged": "overwhelmed",
    "goooood": "good",
    "hubby": "husband",
    "sum": "some",
    "nuggetssss": "nuggets",
    "stinkin": "stinking",
    "furkin": "freaking",
    "eh": "expression of indifference",
    "funfun": "lots of fun",
    "et al": "and others",
    "ala": "American Library Association",
    "uaw": "United Auto Workers",
    "wftb": "waiting for the bus",
    "cox": "internet provider",
    "dslreports": "DSL report website",
    "gt": "greater than",
    "lt": "less than",
    "v": "very",
    "spokenfor": "taken",
    "summize": "Twitter search service",
    "enuf": "enough",
    "pg": "page",
    "kinda": "kind of",
    "receipes": "recipes",
    "lam": "I am",
    "lt": "less than",
    "wysiwyg": "what you see is what you get",
    "texn": "texting",
    "flockofseagullsweregeopoliticallycorrect": "refers to the song 'Iran' by A Flock of Seagulls"
}

contractions = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "it's": "it is",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "let's": "let us",
    "that's": "that is",
    "who's": "who is",
    "what's": "what is",
    "here's": "here is",
    "there's": "there is",
    "where's": "where is",
    "how's": "how is",
    "cant": "can not",
    "wont": "will not",
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
    "havent": "have not",
    "hasnt": "has not",
    "hadnt": "had not",
    "youre": "you are",
    "theyre": "they are",
    "were": "we are",
    "lets": "let us",
    "thats": "that is",
    "whos": "who is",
    "whats": "what is",
    "heres": "here is",
    "theres": "there is",
    "wheres": "where is",
    "hows": "how is",
    "im": "i am",
    " s":"is",
    "ur": "your",
    "u": "you",
    "you ll": "you will",
    "i ve": "I have",
    "it s": "it is",
    "gonna": "going to",
    "wanna": "want to",
    "jus": "just",
    "til": "until",
    "you ll": "you will",
    "i ve": "I have",
    "it s": "it is",
    "gonna": "going to",
    "jus": "just",
    "til": "until",
    "srsly": "seriously",
    "you ll": "you will",
    "won t": "will not",
    "it s": "it is",
    "we ll": "we will",
    "they re": "they are",
    "u": "you",
    "s": "is",
    "amp": "&",
    "twippin": "tripping",
    "mkii": "mark two",
    "ima": "I am going to",
    "dood": "dude",
    "govt": "government",
    "f k": "mess",
    "wftb": "waiting for the bus",
    "gonna": "going to",
    "freakin": "very",
}

# Funciones de preprocesamiento
def replace_abbreviations(text, abbreviation_dict):
    words = text.split()
    return " ".join([abbreviation_dict.get(word.lower(), word) for word in words])

def replace_contractions(text, contractions):
    for contraction, expanded in contractions.items():
        text = re.sub(rf'\b{contraction}\b', expanded, text)
    return text

def preprocess_text(text, abbreviation_dict, contractions):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = replace_abbreviations(text, abbreviation_dict)
    text = replace_contractions(text, contractions)
    return text.strip()

# Paso 1 - Preprocesamiento de texto
data = pd.read_csv('test_data.csv', encoding='latin-1')
data['sentence_original'] = data['sentence']
data['sentence'] = data['sentence'].apply(lambda x: preprocess_text(x, abbreviation_dict, contractions))
data.to_csv(os.path.join(resultados_dir, 'dataset_procesado.csv'), index=False)
print("Paso 1 completado: Datos procesados y guardados en 'Resultados/dataset_procesado.csv'.")

# Paso 2 - Análisis de sentimiento VADER
sia = SentimentIntensityAnalyzer()
df = pd.read_csv(os.path.join(resultados_dir, 'dataset_procesado.csv'))
positive_scores, negative_scores, analysis_times = [], [], []

for text in df['sentence']:
    start_time = time.time()
    scores = sia.polarity_scores(text)
    exec_time = time.time() - start_time
    pos, neg = scores['pos'], scores['neg']
    total = pos + neg
    normalized_pos = pos / total if total > 0 else 0
    normalized_neg = neg / total if total > 0 else 0
    positive_scores.append(normalized_pos)
    negative_scores.append(normalized_neg)
    analysis_times.append(exec_time)

df['positive_score'], df['negative_score'], df['sentiment_analysis_time'] = positive_scores, negative_scores, analysis_times
df.to_csv(os.path.join(resultados_dir, 'dataset_procesado.csv'), index=False)
print("Paso 2 completado: Sentimientos guardados en 'Resultados/dataset_procesado.csv'.")

# Paso 3 - Cálculo de límites de membresía
def calculate_membership_limits(data, score_column):
    min_val = data[score_column].min()
    max_val = data[score_column].max()
    mid_val = (min_val + max_val) / 2
    return min_val, mid_val, max_val

# Función de membresía triangular
def triangular_membership(x, d, e, f):
    if x <= d:
        return 0
    elif d < x <= e:
        return (x - d) / (e - d)
    elif e < x <= f:
        return (f - x) / (f - e)
    else:
        return 0

# Aplicar la fuzzificación
df = pd.read_csv(os.path.join(resultados_dir, 'dataset_procesado.csv'))
d, e, f = calculate_membership_limits(df, 'positive_score')
df['positive_low'] = df['positive_score'].apply(lambda x: triangular_membership(x, 0, d, e))
df['positive_medium'] = df['positive_score'].apply(lambda x: triangular_membership(x, d, e, f))
df['positive_high'] = df['positive_score'].apply(lambda x: triangular_membership(x, e, f, 1))
d, e, f = calculate_membership_limits(df, 'negative_score')
df['negative_low'] = df['negative_score'].apply(lambda x: triangular_membership(x, 0, d, e))
df['negative_medium'] = df['negative_score'].apply(lambda x: triangular_membership(x, d, e, f))
df['negative_high'] = df['negative_score'].apply(lambda x: triangular_membership(x, e, f, 1))
df.to_csv(os.path.join(resultados_dir, 'dataset_procesado.csv'), index=False)
print("Paso 3 completado: Fuzzificación completada y guardada.")

# Paso 4 - Configuración de inferencia difusa
positive_d, positive_e, positive_f = calculate_membership_limits(df, 'positive_score')
negative_d, negative_e, negative_f = calculate_membership_limits(df, 'negative_score')

positive_score = ctrl.Antecedent(np.arange(positive_d, positive_f + 0.1, 0.1), 'positive_score')
negative_score = ctrl.Antecedent(np.arange(negative_d, negative_f + 0.1, 0.1), 'negative_score')
sentiment = ctrl.Consequent(np.arange(0, 10.1, 1), 'sentiment')

positive_score['low'] = fuzz.trimf(positive_score.universe, [0, 0, positive_e])
positive_score['medium'] = fuzz.trimf(positive_score.universe, [positive_d, positive_e, positive_f])
positive_score['high'] = fuzz.trimf(positive_score.universe, [positive_e, positive_f, 1])

negative_score['low'] = fuzz.trimf(negative_score.universe, [0, 0, negative_e])
negative_score['medium'] = fuzz.trimf(negative_score.universe, [negative_d, negative_e, negative_f])
negative_score['high'] = fuzz.trimf(negative_score.universe, [negative_e, negative_f, 1])

sentiment['negative'] = fuzz.trimf(sentiment.universe, [0, 0, 5])
sentiment['neutral'] = fuzz.trimf(sentiment.universe, [0, 5, 10])
sentiment['positive'] = fuzz.trimf(sentiment.universe, [5, 10, 10])

rule1 = ctrl.Rule(positive_score['low'] & negative_score['low'], sentiment['neutral'])
rule2 = ctrl.Rule(positive_score['medium'] & negative_score['low'], sentiment['positive'])
rule3 = ctrl.Rule(positive_score['high'] & negative_score['low'], sentiment['positive'])
rule4 = ctrl.Rule(positive_score['low'] & negative_score['medium'], sentiment['negative'])
rule5 = ctrl.Rule(positive_score['medium'] & negative_score['medium'], sentiment['neutral'])
rule6 = ctrl.Rule(positive_score['high'] & negative_score['medium'], sentiment['positive'])
rule7 = ctrl.Rule(positive_score['low'] & negative_score['high'], sentiment['negative'])
rule8 = ctrl.Rule(positive_score['medium'] & negative_score['high'], sentiment['negative'])
rule9 = ctrl.Rule(positive_score['high'] & negative_score['high'], sentiment['neutral'])

sentiment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
sentiment_simulation = ctrl.ControlSystemSimulation(sentiment_ctrl)

# Función para calcular el sentimiento
def compute_sentiment(pos_score, neg_score):
    sentiment_simulation.input['positive_score'] = pos_score
    sentiment_simulation.input['negative_score'] = neg_score
    sentiment_simulation.compute()
    return sentiment_simulation.output['sentiment']

df['computed_sentiment'] = df.apply(lambda row: compute_sentiment(row['positive_score'], row['negative_score']), axis=1)
df.to_csv(os.path.join(resultados_dir, 'dataset_procesado.csv'), index=False)
print("Paso 4 completado: Inferencia difusa aplicada y guardada.")

# Paso 5 - Clasificación de sentimiento
def classify_sentiment(score):
    if 0 <= score <= 3.3:
        return 'Negative'
    elif 3.3 < score <= 6.7:
        return 'Neutral'
    elif 6.7 < score <= 10:
        return 'Positive'
    else:
        return 'Unknown'

# Aplicar la clasificación y guardar el DataFrame completo
df['sentiment_class'] = df['computed_sentiment'].apply(classify_sentiment)
df.to_csv(os.path.join(resultados_dir, 'dataset_calculos.csv'), index=False)
print("Paso 5 completado: Clasificación de sentimiento guardada en 'Resultados/dataset_calculos.csv'.")

# Paso 6 - Benchmarking de Sentimientos
test_data = pd.read_csv('test_data.csv', encoding='latin-1')
dataset_calculos = pd.read_csv(os.path.join(resultados_dir, 'dataset_calculos.csv'))

# Verificar que la columna 'sentiment_analysis_time' exista en dataset_calculos
if 'sentiment_analysis_time' not in dataset_calculos.columns:
    raise KeyError("La columna 'sentiment_analysis_time' no se encuentra en 'dataset_calculos.csv'. Asegúrate de ejecutar correctamente el paso 2 antes de este paso.")

results = []
execution_times = []

for index, row in dataset_calculos.iterrows():
    original_sentence = test_data.loc[index, 'sentence']
    original_label = test_data.loc[index, 'sentiment']
    exec_time = row['sentiment_analysis_time']
    
    results.append({
        'Oración original': original_sentence,
        'Label original': original_label,
        'Puntaje positivo': row['positive_score'],
        'Puntaje negativo': row['negative_score'],
        'Resultado de inferencia': row['sentiment_class'],
        'Tiempo de ejecución': exec_time
    })
    
    execution_times.append(exec_time)

# Guardar los resultados del benchmark
results_df = pd.DataFrame(results)
output_csv = os.path.join(resultados_dir, 'benchmark_sentiment_results.csv')
results_df.to_csv(output_csv, index=False)
print(f"Paso 6 completado: Benchmark guardado en '{output_csv}'.")

# Calcular el tiempo promedio de ejecución
average_execution_time = np.mean(execution_times)

# Contar los tweets positivos, neutros y negativos
total_positive = len(results_df[results_df['Resultado de inferencia'] == 'Positive'])
total_neutral = len(results_df[results_df['Resultado de inferencia'] == 'Neutral'])
total_negative = len(results_df[results_df['Resultado de inferencia'] == 'Negative'])
total_tweets = len(results_df)

# Calcular los porcentajes
positive_percentage = (total_positive / total_tweets) * 100
neutral_percentage = (total_neutral / total_tweets) * 100
negative_percentage = (total_negative / total_tweets) * 100

# Calcular el tiempo total y promedio
total_execution_time = sum(execution_times)
average_execution_time_per_tweet = total_execution_time / total_tweets

# Imprimir el resumen
print(f"Total de tweets positivos: {total_positive} ({positive_percentage:.2f}%)")
print(f"Total de tweets neutrales: {total_neutral} ({neutral_percentage:.2f}%)")
print(f"Total de tweets negativos: {total_negative} ({negative_percentage:.2f}%)")
print(f"\nTiempo total de ejecución: {total_execution_time:.10f} segundos")
print(f"Tiempo promedio de ejecución por tweet: {average_execution_time_per_tweet:.10f} segundos")
