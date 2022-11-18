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
df = pd.read_csv('Formulador_dataset.csv', on_bad_lines='warn')
df[['Ejercicio', 'Tema']] = df['Ejercicio;Tema'].str.split(';', expand=True)
del df['Ejercicio;Tema']
df.drop_duplicates(inplace=True)

#Creo la raíz
raiz=Tk()

#Defino mis constantes y clases
PADX=10
PADY=10
CLASES = 6 
SIGNIFICADO = {'mru': 0, 'mruv': 1, 'tv': 2, 'tob': 3, 'mcu': 4, 'mcuv': 5}
REDIM = 0.65
mru = Image.open('Ecuaciones/MRU.jpg')
MRU = ImageTk.PhotoImage(mru.resize((round(mru.width*REDIM), round(mru.height*REDIM)), Image.Resampling.LANCZOS))
mruv = Image.open('Ecuaciones/MRUV.jpg')
MRUV = ImageTk.PhotoImage(mruv.resize((round(mruv.width*REDIM), round(mruv.height*REDIM)), Image.Resampling.LANCZOS))
tv = Image.open('Ecuaciones/TV.jpg')
TV = ImageTk.PhotoImage(tv.resize((round(tv.width*REDIM), round(tv.height*REDIM)), Image.Resampling.LANCZOS))
tob = Image.open('Ecuaciones/TOB.jpg')
TOB = ImageTk.PhotoImage(tob.resize((round(tob.width*REDIM), round(tob.height*REDIM)), Image.Resampling.LANCZOS))
mcu = Image.open('Ecuaciones/MCU.jpg')
MCU = ImageTk.PhotoImage(mcu.resize((round(mcu.width*REDIM), round(mcu.height*REDIM)), Image.Resampling.LANCZOS))
mcuv = Image.open('Ecuaciones/MCUV.jpg')
MCUV = ImageTk.PhotoImage(mcuv.resize((round(mcuv.width*REDIM), round(mcuv.height*REDIM)), Image.Resampling.LANCZOS))

IMAGENES = {'mru': mru, 'mruv': mruv, 'tv': tv, 'tob': tob, 'mcu': mcu, 'mcuv': mcuv}
FORMULAS = {'mru': MRU, 'mruv': MRUV, 'tv': TV, 'tob': TOB, 'mcu': MCU, 'mcuv': MCUV}

clase_prob = dict.fromkeys(SIGNIFICADO, 0)

raiz.geometry("1350x650")

raiz.resizable(width=False, height=False)

raiz.title("Formulador Físico")

COLOR = "#333333"

raiz.config(bg=COLOR)

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
miFrame=Frame(raiz, width=1350, height=700)

miFrame.config(bg=COLOR)

miFrame.pack()

leyenda_texto = Label(miFrame, text="Ingrese el problema en la casilla de abajo:", fg = "white", font = ("Calibri", 10), bg=COLOR)

leyenda_texto.place(x=100, y=2)

text_area = ScrolledText(miFrame, wrap = tk.WORD, width = 45, height = 10, font = ("Calibri", 13))
  
text_area.place(x=10, y=25)

#Labels de las imagenes
label1 = Label(miFrame, image='', bg=COLOR)
label1.place(x=10, y= 1300)
label2 = Label(miFrame, image='', bg=COLOR)
label2.place(x=20, y= 1300)
label3 = Label(miFrame, image='', bg=COLOR)
label3.place(x=30, y= 1300)
label4 = Label(miFrame, image='', bg=COLOR)
label4.place(x=40, y= 1300)
label5 = Label(miFrame, image='', bg=COLOR)
label5.place(x=50, y= 1300)
label6 = Label(miFrame, image='', bg=COLOR)
label6.place(x=60, y= 1300)
label7 = Label(miFrame, image='', bg=COLOR)
label7.place(x=70, y= 1300)
label8 = Label(miFrame, image='', bg=COLOR)
label8.place(x=80, y= 1300)
label9 = Label(miFrame, image='', bg=COLOR)
label9.place(x=90, y= 1300)
label10 = Label(miFrame, image='', bg=COLOR)
label10.place(x=100, y= 1300)

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
    #Reseteo los labels y posiciones
    label1.configure(image='')
    label2.configure(image='')
    label3.configure(image='')
    label4.configure(image='')
    label5.configure(image='')
    label6.configure(image='')
    label7.configure(image='')
    label8.configure(image='')
    label9.configure(image='')
    label10.configure(image='')
    label2_x = 0
    label3_x = 0 
    label4_x = 0 
    label5_x = 0 
    label6_x = 0 
    label7_x = 0 
    label8_x = 0
    label9_x = 0
    label10_x = 0


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
    if len(lista_mostrar) >= 1:
        label1.configure(image = FORMULAS[lista_mostrar[0]])
        label1.place(x=PADX, y=300)

    
    if len(lista_mostrar) >= 2:
        label2.configure(image = FORMULAS[lista_mostrar[1]])
        label2_x = round((IMAGENES[lista_mostrar[0]].width)*REDIM)
        label2.place(x=PADX+label2_x, y=300)

    if len(lista_mostrar) >= 3:
        label3.configure(image = FORMULAS[lista_mostrar[2]])
        label3_x = label2_x+round((IMAGENES[lista_mostrar[1]].width)*REDIM)
        label3.place(x=PADX+label3_x, y=320)

    
    if len(lista_mostrar) >= 4:
        label4.configure(image = FORMULAS[lista_mostrar[3]])
        label4_x = label3_x+round((IMAGENES[lista_mostrar[2]].width)*REDIM)
        label4.place(x=PADX+label4_x, y=320)

    if len(lista_mostrar) >= 5:
        label5.configure(image = FORMULAS[lista_mostrar[4]])
        label5_x = label4_x+round((IMAGENES[lista_mostrar[3]].width)*REDIM)
        label5.place(x=PADX+label5_x, y=320)

    if len(lista_mostrar) >= 6:
        label6.configure(image = FORMULAS[lista_mostrar[5]])
        label6_x = 450
        label6.place(x=label6_x, y=10)

    if len(lista_mostrar) >= 7:
        label7.configure(image = FORMULAS[lista_mostrar[6]])
        label7_x = label6_x+round((IMAGENES[lista_mostrar[5]].width)*REDIM)
        label7.place(x=label7_x, y=10)

    if len(lista_mostrar) >= 8:
        label8.configure(image = FORMULAS[lista_mostrar[7]])
        label8_x = label7_x+round((IMAGENES[lista_mostrar[6]].width)*REDIM)
        label8.place(x=label8_x, y=10)

    if len(lista_mostrar) >= 9:
        label9.configure(image = FORMULAS[lista_mostrar[8]])
        label9_x = label8_x+round((IMAGENES[lista_mostrar[7]].width)*REDIM)
        label9.place(x=label9_x, y=10)

    
    if len(lista_mostrar) >= 10:
        label10.configure(image = FORMULAS[lista_mostrar[9]])
        label10_x = label5_x+round((IMAGENES[lista_mostrar[4]].width)*REDIM)
        label10.place(x=PADX+label10_x, y=320)
    
 
    

#Creo el botón de predecir
predecir_but=Button(miFrame, text="Formular", bg="orange2", width=15, height=2, command=lambda:predecir(text_area.get("1.0", tk.END)))
predecir_but.place(x=150, y=250)

raiz.mainloop()
