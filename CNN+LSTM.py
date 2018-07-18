
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal #for openning BIL format image 
from keras.layers import Dense, Conv1D, Input, MaxPooling1D, Flatten, Dropout, LSTM
from keras import Sequential
from keras.utils import np_utils
from PIL import Image
import array
import sys
import os

gdal.GetDriverByName('EHdr').Register()
img = gdal.Open("apex17bands")
b = img.RasterCount
col = img.RasterXSize
row = img.RasterYSize
bands = img.RasterCount
print(col, row)
datatype = gdal.GetDataTypeName(bands)
c_c = int(input("number of class"))
print(datatype)
def train():
    def ReadBilFile(bil,bands,pixels):
        extract_band = 1
        image = np.zeros([pixels, bands], dtype=np.uint16)
        gdal.GetDriverByName('EHdr').Register()
        img = gdal.Open(bil)
        while bands >= extract_band:
            bandx = img.GetRasterBand(extract_band)
            datax = bandx.ReadAsArray()
            temp = datax
            store = temp.reshape(pixels)
            for i in range(pixels):
                image[i][extract_band - 1] = store[i]
            extract_band = extract_band + 1
        return image

    print(os.listdir("train1"))
    path = os.listdir("train1")
    pixels = row * col
    y_test = np.zeros([row * col], dtype=np.uint16)
    x_test = ReadBilFile("apex17bands", bands, pixels)
    x_test = x_test.reshape(row*col, bands, 1)
    values = []
    c_l = {}
    # assigning class to every training file
    for add in path:
        c = int(input())
        print("{} class {} ".format(add, c))
        c_l[add] = c
    clicks={}
    # calculation of clicks in every training file
    for address in path:
        with open('train1'+address, "rb") as f:
            k = len(f.read())
            clicks[address] = (k // 2 // bands) if (k // 2 // bands) < 400 else (k // 2 // bands) // 4
            print('{} ==> {}'.format(address, clicks[address]))
    
    # reading data in training files in binary form
    for address in path:
        with open('train1'+address, "rb") as f:
            b = array.array("H")
            b.fromfile(f, clicks[address]*bands)
            if sys.byteorder == "little":
                b.byteswap()
            for v in b:
                values.append(v)

    ll = (len(values))
    rex = ll // bands
    print(ll, rex)
    
    f_in = np.zeros([ll], dtype=np.uint16)
    x = 0
    for i in range(ll):
        f_in[x] = values[i]
        x += 1

    y_train = np.zeros([rex], dtype=np.uint16)
    
    mark = 0
    for add in path:
        for i in range(clicks[add]):
            y_train[mark+i] = c_l[add]
        mark = mark + clicks[add]

        
    x_train = f_in.reshape(rex, bands)
    print(x_train)

    seed = 7
    np.random.seed(seed)

    x_train = x_train / 2**16-1
    x_test = x_test / 2**16-1
    num_pixels = bands

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = c_c
    y_test_new = np.zeros([pixels, c_c], dtype=np.uint16)

    print(x_test)
    print(20*'#')
    print(x_train)
    print(20*'#')
    print(y_test)
    print(20*'#')
    print(y_train)

    print(x_test.shape)
    print(x_train.shape)
    print(y_train.shape)
    print(y_test.shape)

    X = x_train.reshape(x_train.shape[0], bands, 1)

    model = Sequential()
    model.add(Conv1D(2 ** 3, 2, activation="relu", padding='same', input_shape=[17, 1]))
    model.add(LSTM(2**4, return_sequences=True))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(2 ** 5, 2, activation="relu", padding='same'))
    model.add(LSTM(2**6, return_sequences=True))
    model.add(MaxPooling1D(2))
    model.add(LSTM(2**8, return_sequences=True))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X, y_train, batch_size=50, epochs=3000)
    y_test_new = model.predict(x_test, batch_size=50)
    print(y_test_new.shape)
    y_test1 = np.argmax(y_test_new, axis=1)
    print("this is predicted output")

    k = y_test1.reshape(row, col)
    plt.imshow(k)
    plt.show()
    result = Image.fromarray((k * (2**16-1)//c_c).astype('uint16'))
    result.save('Classified_images_12/hard.tiff')
    
    try:
        os.mkdir("Classified_images_12")
    except:
        pass
    
    for i in range(1, 8):
        img = y_test_new[:,i].reshape(row, col)
        plt.imshow(img*(2**16-1))
        plt.colorbar()
        plt.show()
        result = Image.fromarray(((img * (2**16-1))).astype('uint16'))
        result.save('Classified_images_12/1_'+str(i)+'_other.tiff')


train()

