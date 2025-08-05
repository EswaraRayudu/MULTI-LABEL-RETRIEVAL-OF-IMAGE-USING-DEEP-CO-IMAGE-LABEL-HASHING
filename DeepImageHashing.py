from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import cv2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import model_from_json
import string
from keras.models import model_from_json
from keras.layers import Dense, Activation, LSTM, Embedding, TimeDistributed, RepeatVector


main = tkinter.Tk()
main.title("Multi-label Retrieval of Image Using Deep Co-Image-Label Hashing")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test
global model
global filename
global X, Y
image_hash = []
image_label = []
global image_text_tokenized, image_text_tokenizer, label_text_tokenized, label_text_tokenizer, image_vocab, label_vocab
global max_image_len, max_label_len, image_pad_sentence, label_pad_sentence, label_pad_sentence
global enc_dec_model

def hash_array_to_hash_hex(hash_array):      # convert hash array of 0 or 1 to hash string in hex
  hash_array = np.array(hash_array, dtype = np.uint8)
  hash_str = ''.join(str(i) for i in 1 * hash_array.flatten())
  return (hex(int(hash_str, 2)))

def hash_hex_to_hash_array(hash_hex):      # convert hash string in hex to hash values of 0 or 1
  hash_str = int(hash_hex, 16)
  array_str = bin(hash_str)[2:]
  return np.array([i for i in array_str], dtype = np.float32)

def getHash(name):
    img = cv2.imread(name)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img, dtype = np.float32)
    dct = cv2.dct(img)
    dct_block = dct[: 8, : 8]
    dct_average = (dct_block.mean() * dct_block.size - dct_block[0, 0]) / (dct_block.size - 1)
    dct_block[dct_block < dct_average] = 0.0
    dct_block[dct_block != 0] = 1.0
    hashing = hash_array_to_hash_hex(dct_block.flatten())
    return hashing.strip()

def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")

def clean_sentence(sentence):
    lower_case_sent = sentence.lower()
    string_punctuation = string.punctuation + "¡" + '¿'
    clean_sentence = lower_case_sent.translate(str.maketrans('', '', string_punctuation))
    return clean_sentence

def tokenize(sentences):
    text_tokenizer = Tokenizer()
    text_tokenizer.fit_on_texts(sentences)
    return text_tokenizer.texts_to_sequences(sentences), text_tokenizer

def preprocessDataset():
    text.delete('1.0', END)
    global X, Y
    global image_hash, image_label
    global image_text_tokenized, image_text_tokenizer, label_text_tokenized, label_text_tokenizer, image_vocab, label_vocab
    global max_image_len, max_label_len, image_pad_sentence, label_pad_sentence, label_pad_sentence
    image_hash.clear()
    image_label.clear()
    dup = []
    with open("model/captions.txt", "r") as file:
        for line in file:
            line = line.strip('\n')
            line = line.strip()
            arr = line.split(",")
            if arr[0] != 'image' and len(image_hash) <= 130 and arr[0] not in dup:
                dup.append(arr[0])
                caption = arr[1].strip()
                image_label.append(caption.strip())
                words = getHash("Dataset/Images/"+arr[0])
                image_hash.append(words)
    file.close()
    image_text_tokenized, image_text_tokenizer = tokenize(image_hash)
    label_text_tokenized, label_text_tokenizer = tokenize(image_label)
    image_vocab = len(image_text_tokenizer.word_index) + 1
    label_vocab = len(label_text_tokenizer.word_index) + 1
    max_image_len = int(len(max(image_text_tokenized,key=len)))
    max_label_len = int(len(max(label_text_tokenized,key=len)))

    image_pad_sentence = pad_sequences(image_text_tokenized, max_image_len, padding = "post")
    label_pad_sentence = pad_sequences(label_text_tokenized, max_label_len, padding = "post")
    label_pad_sentence = label_pad_sentence.reshape(*label_pad_sentence.shape, 1)
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    text.insert(END,"Image & labels processing completed\n")
    test = X[3]
    cv2.imshow("Processed Image",cv2.resize(test,(400,400)))
    cv2.waitKey(0)

def trainDCILH():
    global enc_dec_model
    text.delete('1.0', END)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            enc_dec_model = model_from_json(loaded_model_json)
        json_file.close()
        enc_dec_model.load_weights("model/model_weights.h5")
    else:
        input_sequence = Input(shape=(max_image_len,))
        embedding = Embedding(input_dim=image_vocab, output_dim=128)(input_sequence)
        encoder = LSTM(32, return_sequences=False)(embedding)
        r_vec = RepeatVector(max_label_len)(encoder)
        decoder = LSTM(32, return_sequences=True, dropout=0.2)(r_vec)
        logits = TimeDistributed(Dense(label_vocab))(decoder)
        enc_dec_model = Model(input_sequence, Activation('softmax')(logits))
        enc_dec_model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(1e-3), metrics=['accuracy'])
        enc_dec_model.summary()
        hist = enc_dec_model.fit(image_pad_sentence, label_pad_sentence, batch_size=4, epochs=2000)
        enc_dec_model.save('model/full_model.h5')
        model_json = enc_dec_model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)

        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    X_train, X_test, y_train, y_test = train_test_split(image_pad_sentence, label_pad_sentence, test_size=0.2, random_state=0)
    predict = enc_dec_model.predict(X_test)
    predicts = []
    y_tests = []
    for i in range(0, predict.shape[0]):
        for j in range(0, predict.shape[1]):
            value = np.argmax(predict[i, j])
            predicts.append(value)
            y_tests.append(y_test[i, j, 0])
    p = precision_score(y_tests, predicts, average='macro') * 100
    r = recall_score(y_tests, predicts, average='macro') * 100
    f = f1_score(y_tests, predicts, average='macro') * 100
    a = accuracy_score(y_tests, predicts) * 100
    text.insert(END, 'DCILH Accuracy  : ' + str(a) + "\n")
    text.insert(END, 'DCILH Precision : ' + str(p) + "\n")
    text.insert(END, 'DCILH Recall    : ' + str(r) + "\n")
    text.insert(END, 'DCILH FScore    : ' + str(f) + "\n")


def predictLabel(logits, tokenizer):
    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = '' 
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def predict():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    testCode_tokenize = image_text_tokenizer.texts_to_sequences([getHash(filename)])
    testCode_tokenize = pad_sequences(testCode_tokenize, max_image_len, padding = "post")
    predict_label = predictLabel(enc_dec_model.predict(testCode_tokenize)[0], label_text_tokenizer)
    text.insert(END,"Predicted Multi label from given image: "+str(predict_label))
    text.update_idletasks()
    img = cv2.imread(filename)
    img = cv2.resize(img, (1200,600))
    cv2.putText(img, 'Predicted labels : '+str(predict_label), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Predicted labels : '+str(predict_label), img)
    cv2.waitKey(0)
                
def close():
    main.destroy()

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['accuracy']
    loss = data['loss']
    '''
    accuracy = accuracy[4500:4999]
    loss = loss[4500:4999]
    '''
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('DCILH Training Accuracy & Loss Graph')
    plt.show() 

font = ('times', 14, 'bold')
title = Label(main, text='Multi-label Retrieval of Image Using Deep Co-Image-Label Hashing')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=138)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Flickr Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess Labels & Images", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

trainButton = Button(main, text="Train DCILH Model", command=trainDCILH)
trainButton.place(x=50,y=200)
trainButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=50,y=250)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Multi Labels from Image", command=predict)
predictButton.place(x=50,y=300)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=350)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='teal')
main.mainloop()