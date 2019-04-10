import os
import numpy as np
import cv2
from pathlib import Path
import keras
import keras.utils
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import matplotlib.pyplot as plt


TRAIN_PATH = Path.cwd() / 'cad_pictures' / 'train'
TEST_PATH = Path.cwd() / 'cad_pictures' / 'test'
MODEL_PATH = Path.cwd() / 'saved_models'


ROWS = COLS = 128
CHANNELS = 1
learning_rate = 0.0001
nb_epoch = 50
batch_size = 64
NUM_IMAGES = 25000  # max: 25000
FLOAT_TYPE = 'float32'
K.set_image_data_format('channels_first')
K.set_floatx(FLOAT_TYPE)


# Read input path
train_images = np.array([TRAIN_PATH / i for i in os.listdir(TRAIN_PATH)])
test_images = np.array([TEST_PATH / i for i in os.listdir(TEST_PATH)])
train_cats = np.array([TRAIN_PATH / i for i in os.listdir(TRAIN_PATH) if 'cat' in i])
train_dogs = np.array([TRAIN_PATH / i for i in os.listdir(TRAIN_PATH) if 'dog' in i])

# Subsample the inputs
train_images = np.append(train_cats[:int(NUM_IMAGES/2)], train_dogs[:int(NUM_IMAGES/2)])
np.random.shuffle(train_images)
test_images = test_images[:100]
print(train_images.shape)
print(test_images.shape)


# Read images, split between train and test sets, normalize the data
def read_image(file_path):
    if CHANNELS == 3:
        img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_LANCZOS4)


def prep_data(images):
    count = images.shape[0]
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i % 250 == 0: print('Processed {} of {}'.format(i, count))
    return data


train = prep_data(train_images)
test = prep_data(test_images)
train = train.astype(FLOAT_TYPE)
test = test.astype(FLOAT_TYPE)
train /= 255
test /= 255

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))

labels = []
for i in range(train_images.shape[0]):
    if 'cat' in train_images[i].name:
        labels.append(1)
    else:
        labels.append(0)


# Set up the network
def catdog():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(CHANNELS, ROWS, COLS), activation='relu',))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',))
    model.add(MaxPooling2D(pool_size=(3, 3),))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',))
    model.add(MaxPooling2D(pool_size=(3, 3),))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',))
    model.add(MaxPooling2D(pool_size=(3, 3),))

    model.add(Flatten())
    model.add(Dense(units=256, activation='relu',))
    model.add(Dropout(0.3))
    model.add(Dense(units=256, activation='relu',))
    model.add(Dropout(0.3))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    keras.utils.print_summary(model)

    return model


model = catdog()


# Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


history = LossHistory()

# Set up callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto', restore_best_weights=True)
check_path = MODEL_PATH / 'cats_and_dogs.hdf5'
checkpoint = ModelCheckpoint(os.fspath(check_path), monitor='val_acc', verbose=1, save_best_only=True, mode='auto')


# Start training
model.fit(train, labels, batch_size=batch_size, epochs=nb_epoch, validation_split=0.10,
          verbose=2, shuffle=True, callbacks=[history, early_stopping, checkpoint, reduce_lr])
predictions = model.predict(test, verbose=1)


# Plot losses during the training
loss = history.losses
val_loss = history.val_losses
cv2.waitKey(0)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0, nb_epoch)[0::2])
plt.legend()
plt.show()

# Predict some images
for i in range(0, 5):
    if predictions[i, 0] >= 0.5:
        print('I am {:.2%} sure this is a Cat'.format(predictions[i][0]))
    else:
        print('I am {:.2%} sure this is a Dog'.format(1 - predictions[i][0]))
    if CHANNELS == 3:
        plt.imshow(np.array(test[i].T.astype('float32')))
    else:
        plt.imshow(np.reshape(np.array(test[i].T.astype('float32')), (ROWS, COLS)), cmap='gray')
    plt.show()
