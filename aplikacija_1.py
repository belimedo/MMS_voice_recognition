import os
import sys
import random
import time
import string
import pickle
import warnings
import numpy as np
import sounddevice as sd 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # za uklanjanje TF upozorenja
import tensorflow as tf

from math import ceil
from tqdm import tqdm
from scipy.io.wavfile import write
from scipy.io.wavfile import read
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
from sklearn.model_selection import train_test_split # novo
from sklearn.model_selection import cross_val_score # novo
from sklearn.model_selection import ShuffleSplit # novo

from utils_application import wavefile_to_waveform, prepareForClasification, replay_words

import openl3
try:
    import openl3
except:
    print('Warning: you did not install openl3, you cannot use this feature extractor (but you can use the pre-computed features).')


def extractWordsWindowAppendable(array,thd = 0.05,window_time = 0.05,fs = 22050):
    
    TRESHHOLD_VALUE = max(array)*thd
    WINDOW_SIZE = int(np.floor(fs*window_time))
    ACCEPTABLE_SIZE = WINDOW_SIZE * 0.2
    MAX_SKIPPED = 2
    b = len(array)//WINDOW_SIZE
    rest = b*WINDOW_SIZE
    words = []
    word = np.zeros(0, dtype = np.int16)
    temp_word = np.zeros(0, dtype = np.int16)
    window_num = 0
    skipped = 0
    for i in range(b):
        
        # ako je broj odmjeraka u prozoru koji imaju amplitudu vecu od thd veci od pola dodaj na word vektor i povecaj broj prozora
        if (len(array[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE][np.abs(array[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE]) > TRESHHOLD_VALUE]) > ACCEPTABLE_SIZE):
            
            # ako je preskocen jedan prozor, dodaj ga na rijec
            if (skipped != 0):
                word = np.concatenate((word,temp_word))
                skipped = 0
                temp_word = np.zeros(0, dtype = np.int16)
    
            word = np.concatenate((word,array[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE]))
            window_num += 1  
        
        # ako nije zadovoljen uslov, udji u skipped prvo 2 puta
        elif (skipped < MAX_SKIPPED):
            skipped += 1
            temp_word = np.concatenate((temp_word,array[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE]))
            
        # ako je broj prozora veci od 0 i potroseni su svi skipped, to je sada rijec i trebamo je dodati u listu te resetovati rijec i brojac
        elif (window_num > 0):
            # pitanje da li je ovo ok
            if (skipped > 0):
                word = np.concatenate((word,temp_word))
            words.append(word)
            window_num = 0
            skipped = 0
            word = np.zeros(0, dtype = np.int16)
            temp_word = np.zeros(0, dtype = np.int16)
            
    # kada prodjemo kroz sve prozore, ostaje jos odmjeraka koji nisu ispitani, pored toga ostaje nam i potencijalna posljednja rijec
    # zbog toga prvo ispitujemo da li imamo potencijalnu rijec, ukoliko je nemamo ne trebamo ni provjeravati posljednji prozor
    # trajanja kraceg od 10ms
    if (window_num > 0):
        if (len(array[rest:len(array)][np.abs(array[rest:len(array)]) > TRESHHOLD_VALUE]) > (len(array)-rest) * 0.5):
            word = np.concatenate((word,array[rest:len(array)]))
        words.append(word)
    
    return words


def izgovori_komandu():

    fs = 22050
    window_time = 0.08
    duration = 4

    pripremljene = []

    if not os.path.exists('./temp'):
        os.mkdir("temp")

    time.sleep(0.2)
    print("Počnite govor:")
    recording = sd.rec(int(duration*fs),samplerate=fs,channels=1)
    sd.wait()
    print("Snimanje završeno.")
    # Potrebno je snimiti i ponovo učitati signal, te koristiti vrijednosti iz vektora data da bismo obezbjedili 
    # ispravno funkcionisanje programa
    start = time.time()
    write('./temp/glas_sleep.wav',fs,recording)
    fs, data = read('./temp/glas_sleep.wav')
    os.remove('./temp/glas_sleep.wav') # brisemo fajl
    stop = time.time()
    print("Vrijeme potroseno za snimanje, ucitavanje i brisanje govora {}".format(np.round(stop-start,7)))
    
    start = time.time()
    words = extractWordsWindowAppendable(data,thd = 0.05,window_time = window_time,fs = fs)
    stop = time.time()
    print("Vrijeme potroseno za izvlacenje rijeci iz snimka {}".format(np.round(stop-start,7)))

    if len(words) < 3:
        print("Prepoznao je {}, ponovni pokusaj izvlacenja rijeci...".format(len(words)))
        start = time.time()
        words = extractWordsWindowAppendable(data,thd = 0.05,window_time = 0.06,fs = fs)
        stop = time.time()
        print("Vrijeme potroseno za ponovno izvlacenje rijeci iz snimka {}".format(np.round(stop-start,7)))
    # if len(words) < 3:
    #     print("Prebrzo ste izgovorili rijeci, pokusajte napraviti kratku pauzu izmedju rijeci (bar 100 ms).")
    #     return None,None

    start = time.time()
    for w in words:
        pripremljene.append(prepareForClasification(w,fs*duration))
    stop = time.time()
    print("Vrijeme potroseno za spremanje rijeci za klasifikaciju {}".format(np.round(stop-start,7)))
    return words, pripremljene

def extract_one_openl3_feature(path):
    
    model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music", embedding_size=512)

    wave, sr = wavefile_to_waveform(path, 'openl3')
    emb, _ = openl3.get_audio_embedding(wave, sr, hop_size=1, model=model, verbose=False)

    return emb

def find_predictions(model1,model2, words,fs = 22050):

    if not os.path.exists('./temp'):
        os.mkdir('temp')
    
    command = []
    
    start = time.time()

    for i in range(len(words)):
        
        feature = extract_one_openl3_feature('./temp/word_{}'.format(i))
        pred1 = model1.predict(feature)
        pred2 = model2.predict(feature)
        # Spojimo dvije predikcije
        pred = np.concatenate((pred1,pred2))
        # Pronadjemo koju rijec prikazuje
        command.append(word_labels[np.argmax(np.bincount(pred))])
    
    stop = time.time()
    print("Vrijeme potroseno za izvlacenje ekstrakciju obiljezja {}".format(stop-start))

    return command        


word_labels = {
    0: 'grijanje',
    1: 'hladjenje',
    2: 'iskljuci',
    3: 'kuco',
    4: 'otvori',
    5: 'prozor', 
    6: 'svjetlo',
    7: 'ukljuci',
    8: 'vrata',
    9: 'zatvori'
}

if __name__ == '__main__':

    
    warnings.filterwarnings("ignore") # Da se ne vide upozorenja od TF kada se buni zbog zauzeca radne memorije prilikom izvlacenja obiljezja

    # Ucitavamo dva modela, sa rbf i poly kernelom SVM 
    if os.path.exists('models_final/all_data_kernelpoly_data_representable_03_openl3.pkl'):
        with open('models_final/all_data_kernelpoly_data_representable_03_openl3.pkl', 'rb') as fid:
            model = pickle.load(fid)

    if os.path.exists('models_final/all_data_kernelrbf_data_representable_03_openl3.pkl'):
        with open('models_final/all_data_kernelrbf_data_representable_03_openl3.pkl', 'rb') as fid:
            model2 = pickle.load(fid)


    menu = {}
    old_words = None
    old_command = None
    formated_words = None
    fs = 22050
    menu['1']="Izgovori komandu" 
    menu['2']="Poslušaj posljednju izgovorenu komandu"
    menu['3']="Izađi iz aplikacije"

    while True:
        options=list(menu.keys())
        options.sort()
        for entry in options: 
            print (entry, menu[entry])
        selection=input("Izaberi opciju:\n") 
        if selection =='1': 
            old_words, formated_words = izgovori_komandu()
            if formated_words:
                print("Upisivanje izgovorenih rijeci...")
                for i in range(len(formated_words)):
                    write('./temp/word_{}'.format(i),fs,formated_words[i])

                command = find_predictions(model,model2,formated_words)
                old_command = ""
                for c in command:
                    old_command = old_command + c + " "
                print("Broj izdvojenih rijeci je: {}".format(len(command)))
                print("\n\nIzgovorena komanda je: ")
                for c in command:
                    print(c)

                for path, dirs, files in os.walk('./temp'):
                    for f in files:
                        os.remove(os.path.join(path,f))
                
        elif selection == '2': 
            if old_words == None:
                print("Ne postoji prethodno izgovorena komanda. \n")
            else:
                print("Prethodna komanda: {}\n".format(old_command))
                replay_words(old_words)

        elif selection == '3':
            print("Izlazak iz aplikacije")
            os.rmdir('./temp')
            break
        else: 
            print("Nepoznata komanda") 