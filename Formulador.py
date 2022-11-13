#Importo librerías necesarias para la interfaz
from tkinter import Tk, Frame, Label, Button
from tkinter.scrolledtext import ScrolledText
import tkinter as tk
from PIL import ImageTk, Image


#Importo librearías para preprocesamiento
import nltk
nltk.download('stopwords')
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
#from numpy import argmax

#Importo el modelo y tensorflow
import tensorflow as tf
modelo = tf.keras.models.load_model('modelo_GRU')

#Importo el dataset para el Countvectorizer y tdidf
df = pd.read_csv('Solucionador_dataset.csv', on_bad_lines='warn')
df[['Ejercicio', 'Tema']] = df['Ejercicio;Tema'].str.split(';', expand=True)
del df['Ejercicio;Tema']
df.drop_duplicates(inplace=True)

#Creo la raíz
raiz=Tk()

#Defino mis constantes y clases
CLASES = 6 
SIGNIFICADO = {'mru': 0, 'mruv': 1, 'tv': 2, 'tob': 3, 'mcu': 4, 'mcuv': 5}
REDIM = 0.75
mru = Image.open('Ecuaciones/MRU.jpg')
MRU = ImageTk.PhotoImage(mru.resize((round(mru.width*REDIM), round(mru.height*REDIM)), Image.Resampling.LANCZOS))
mruv = Image.open('Ecuaciones/MRUV.jpg')
MRUV = ImageTk.PhotoImage(mruv.resize((round(mru.width*REDIM), round(mru.height*REDIM)), Image.Resampling.LANCZOS))
tv = Image.open('Ecuaciones/TV.jpg')
TV = ImageTk.PhotoImage(tv.resize((round(mru.width*REDIM), round(mru.height*REDIM)), Image.Resampling.LANCZOS))
tob = Image.open('Ecuaciones/TOB.jpg')
TOB = ImageTk.PhotoImage(tob.resize((round(mru.width*REDIM), round(mru.height*REDIM)), Image.Resampling.LANCZOS))
"""mcu = Image.open('Ecuaciones/MRU.jpg')
MCU = mru.resize((round(mru.width*0.8), round(mru.height*0.8)), Image.Resampling.LANCZOS)"""

FORMULAS = {'mru': MRU, 'mruv': MRUV, 'tv': TV, 'tob': TOB, 'mcu': [], 'mcuv': []}
clase_prob = dict.fromkeys(SIGNIFICADO, 0)

raiz.geometry("800x600")

raiz.resizable(width=False, height=False)

raiz.title("Solucionador de problemas")

color = "#333333"

raiz.config(bg=color)

raiz.iconbitmap("LOGO.ico")

#Elimino caracteres especiales
special_character_remover = re.compile('[/(){}\[\]\|@,;]')
extra_symbol_remover = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('spanish'))


#Funcion limpiar texto
def clean_text(text):
    text = text.lower()
    text = special_character_remover.sub(' ', text)
    text = extra_symbol_remover.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text
    
df['Ejercicio'] = df['Ejercicio'].apply(clean_text)

#Entreno el CountVectorizer y TfIDF
vectorizer=TfidfVectorizer(min_df=1, max_df=0.9, decode_error='warn')
vectorized=vectorizer.fit_transform(df.Ejercicio)

count_vectorizer=CountVectorizer(min_df=1, max_df=0.9, decode_error='warn')
count_vectorized=count_vectorizer.fit_transform(df.Ejercicio)

#Creo el Frame principal
miFrame=Frame(raiz, width=800, height=600)

miFrame.config(bg=color)

miFrame.pack()

leyenda_texto = Label(miFrame, text="Ingrese el problema en la casilla de abajo:", fg = "white", font = ("Calibri", 10), bg=color)

leyenda_texto.grid(row=0, column=0, pady=2)

text_area = ScrolledText(miFrame, wrap = tk.WORD, width = 45, height = 10, font = ("Calibri", 13))
  
text_area.grid(row = 1, column = 0, columnspan=5, pady=3)

#Función transformar texto, recibe como parametro el problema en pantalla
def predecir(problema):
    #Preproceso el problema para disminuir el error
    problema = problema.replace("ñ", "n")
    problema = problema.replace("á", "a")
    problema = problema.replace("é", "e")
    problema = problema.replace("í", "i")
    problema = problema.replace("ó", "o")
    problema = problema.replace("ú", "u")
    problema = problema.replace("ü", "u")
    problema = problema.replace("/", " ")
    problema = problema.replace("\n", "")
    #Uso la función de limpiar texto y creo un solo vector de datos
    lista = [clean_text(problema)]
    lista_tf = vectorizer.transform(lista)
    lista_cont = count_vectorizer.transform(lista)
    lista = sparse.hstack([lista_tf, lista_cont])
    lista = lista.toarray()
    lista = lista.reshape(lista.shape[0], 1,lista.shape[1])
    logit = tf.nn.softmax(modelo.predict(lista))
    logit_mos = tf.reshape(logit, [-1])
    mostrar(logit_mos.numpy())
    """y_pred = argmax(logit, axis=1)
    print(y_pred)"""


def mostrar(pred):
    cont_lis = 0
    cont_mos = 0
    lista_mostrar = []

    for i in clase_prob.keys():
        clase_prob[i] = pred[cont_lis]
        cont_lis += 1
    print(clase_prob)
    #Ordeno de menor a mayor las probabilidades
    clase_prob_ord_inver = dict(sorted(clase_prob.items(), key=lambda item: item[1]))
    #Invierto el orden para que sea el deseado
    clase_prob_ord = dict(reversed(list(clase_prob_ord_inver.items())))
    print(clase_prob_ord)    
    for i in clase_prob_ord.keys():
        if cont_mos>=10:
            break
        if clase_prob_ord[i] > 1/CLASES:
            lista_mostrar.append(i)
        cont_mos += 1
    print(lista_mostrar)
    
    #Dependiendo de cuantas respuestas haya imprimó las ecuaciones características
    if len(lista_mostrar) == 1:
        label1 = Label(miFrame, image = FORMULAS[lista_mostrar[0]])
        label1.grid(row = 3, column = 0)
        
    elif len(lista_mostrar) == 2:
        label1 = Label(miFrame, image = FORMULAS[lista_mostrar[0]])
        label1.grid(row = 3, column = 0)
        label2 = Label(miFrame, image = FORMULAS[lista_mostrar[1]])
        label2.grid(row = 3, column = 1)

    elif len(lista_mostrar) == 3:
        label1 = Label(miFrame, image = FORMULAS[lista_mostrar[0]])
        label1.grid(row = 3, column = 0)
        label2 = Label(miFrame, image = FORMULAS[lista_mostrar[1]])
        label2.grid(row = 3, column = 1)
        label3 = Label(miFrame, image = FORMULAS[lista_mostrar[2]])
        label3.grid(row = 3, column = 2)

    elif len(lista_mostrar) == 4:
        label1 = Label(miFrame, image = FORMULAS[lista_mostrar[0]])
        label1.grid(row = 3, column = 0)
        label2 = Label(miFrame, image = FORMULAS[lista_mostrar[1]])
        label2.grid(row = 3, column = 1)
        label3 = Label(miFrame, image = FORMULAS[lista_mostrar[2]])
        label3.grid(row = 3, column = 2)
        label4 = Label(miFrame, image = FORMULAS[lista_mostrar[3]])
        label4.grid(row = 3, column = 3)

    elif len(lista_mostrar) == 5:
        label1 = Label(miFrame, image = FORMULAS[lista_mostrar[0]])
        label1.grid(row = 3, column = 0)
        label2 = Label(miFrame, image = FORMULAS[lista_mostrar[1]])
        label2.grid(row = 3, column = 1)
        label3 = Label(miFrame, image = FORMULAS[lista_mostrar[2]])
        label3.grid(row = 3, column = 2)
        label4 = Label(miFrame, image = FORMULAS[lista_mostrar[3]])
        label4.grid(row = 3, column = 3)
        label5 = Label(miFrame, image = FORMULAS[lista_mostrar[4]])
        label5.grid(row = 3, column = 4)

    elif len(lista_mostrar) == 6:
        label1 = Label(miFrame, image = FORMULAS[lista_mostrar[0]])
        label1.grid(row = 3, column = 0)
        label2 = Label(miFrame, image = FORMULAS[lista_mostrar[1]])
        label2.grid(row = 3, column = 1)
        label3 = Label(miFrame, image = FORMULAS[lista_mostrar[2]])
        label3.grid(row = 3, column = 2)
        label4 = Label(miFrame, image = FORMULAS[lista_mostrar[3]])
        label4.grid(row = 3, column = 3)
        label5 = Label(miFrame, image = FORMULAS[lista_mostrar[4]])
        label5.grid(row = 3, column = 4)
        label6 = Label(miFrame, image = FORMULAS[lista_mostrar[5]])
        label6.grid(row = 0, column = 1)

    elif len(lista_mostrar) == 7:
        label1 = Label(miFrame, image = FORMULAS[lista_mostrar[0]])
        label1.grid(row = 3, column = 0)
        label2 = Label(miFrame, image = FORMULAS[lista_mostrar[1]])
        label2.grid(row = 3, column = 1)
        label3 = Label(miFrame, image = FORMULAS[lista_mostrar[2]])
        label3.grid(row = 3, column = 2)
        label4 = Label(miFrame, image = FORMULAS[lista_mostrar[3]])
        label4.grid(row = 3, column = 3)
        label5 = Label(miFrame, image = FORMULAS[lista_mostrar[4]])
        label5.grid(row = 3, column = 4)
        label6 = Label(miFrame, image = FORMULAS[lista_mostrar[5]])
        label6.grid(row = 0, column = 1)
        label7 = Label(miFrame, image = FORMULAS[lista_mostrar[6]])
        label7.grid(row = 0, column = 2)

    elif len(lista_mostrar) == 8:
        label1 = Label(miFrame, image = FORMULAS[lista_mostrar[0]])
        label1.grid(row = 3, column = 0)
        label2 = Label(miFrame, image = FORMULAS[lista_mostrar[1]])
        label2.grid(row = 3, column = 1)
        label3 = Label(miFrame, image = FORMULAS[lista_mostrar[2]])
        label3.grid(row = 3, column = 2)
        label4 = Label(miFrame, image = FORMULAS[lista_mostrar[3]])
        label4.grid(row = 3, column = 3)
        label5 = Label(miFrame, image = FORMULAS[lista_mostrar[4]])
        label5.grid(row = 3, column = 4)
        label6 = Label(miFrame, image = FORMULAS[lista_mostrar[5]])
        label6.grid(row = 0, column = 1)
        label7 = Label(miFrame, image = FORMULAS[lista_mostrar[6]])
        label7.grid(row = 0, column = 2)
        label8 = Label(miFrame, image = FORMULAS[lista_mostrar[7]])
        label8.grid(row = 1, column = 1)

    elif len(lista_mostrar) == 9:
        label1 = Label(miFrame, image = FORMULAS[lista_mostrar[0]])
        label1.grid(row = 3, column = 0)
        label2 = Label(miFrame, image = FORMULAS[lista_mostrar[1]])
        label2.grid(row = 3, column = 1)
        label3 = Label(miFrame, image = FORMULAS[lista_mostrar[2]])
        label3.grid(row = 3, column = 2)
        label4 = Label(miFrame, image = FORMULAS[lista_mostrar[3]])
        label4.grid(row = 3, column = 3)
        label5 = Label(miFrame, image = FORMULAS[lista_mostrar[4]])
        label5.grid(row = 3, column = 4)
        label6 = Label(miFrame, image = FORMULAS[lista_mostrar[5]])
        label6.grid(row = 0, column = 1)
        label7 = Label(miFrame, image = FORMULAS[lista_mostrar[6]])
        label7.grid(row = 0, column = 2)
        label8 = Label(miFrame, image = FORMULAS[lista_mostrar[7]])
        label8.grid(row = 1, column = 1)
        label9 = Label(miFrame, image = FORMULAS[lista_mostrar[8]])
        label9.grid(row = 1, column = 2)

    elif len(lista_mostrar) == 10:
        label1 = Label(miFrame, image = FORMULAS[lista_mostrar[0]])
        label1.grid(row = 3, column = 0)
        label2 = Label(miFrame, image = FORMULAS[lista_mostrar[1]])
        label2.grid(row = 3, column = 1)
        label3 = Label(miFrame, image = FORMULAS[lista_mostrar[2]])
        label3.grid(row = 3, column = 2)
        label4 = Label(miFrame, image = FORMULAS[lista_mostrar[3]])
        label4.grid(row = 3, column = 3)
        label5 = Label(miFrame, image = FORMULAS[lista_mostrar[4]])
        label5.grid(row = 3, column = 4)
        label6 = Label(miFrame, image = FORMULAS[lista_mostrar[5]])
        label6.grid(row = 0, column = 1)
        label7 = Label(miFrame, image = FORMULAS[lista_mostrar[6]])
        label7.grid(row = 0, column = 2)
        label8 = Label(miFrame, image = FORMULAS[lista_mostrar[7]])
        label8.grid(row = 1, column = 1)
        label9 = Label(miFrame, image = FORMULAS[lista_mostrar[8]])
        label9.grid(row = 1, column = 2)
        label10 = Label(miFrame, image = FORMULAS[lista_mostrar[9]])
        label10.grid(row = 3, column = 5)
 
    

#Creo el botón de predecir
predecir_but=Button(miFrame, text="Formular", bg="orange2", width=15, height=2, command=lambda:predecir(text_area.get("1.0", tk.END)))
predecir_but.grid(row=2, column=0, pady=5, padx=5)

raiz.mainloop()