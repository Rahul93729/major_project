# import numpy as np
# import argparse
# import cv2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os
# import matplotlib.pyplot as plt

# # Suppress TensorFlow debugging logs and oneDNN warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# # Command-line argument for mode selection
# ap = argparse.ArgumentParser()
# ap.add_argument("--mode", help="train/display")
# a = ap.parse_args()
# mode = a.mode

# def plot_model_history(model_history):
#     fig, axs = plt.subplots(1, 2, figsize=(15, 5))
#     axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
#     axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
#     axs[0].set_title('Model Accuracy')
#     axs[0].set_ylabel('Accuracy')
#     axs[0].set_xlabel('Epoch')
#     axs[0].legend(['train', 'val'], loc='best')

#     axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
#     axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
#     axs[1].set_title('Model Loss')
#     axs[1].set_ylabel('Loss')
#     axs[1].set_xlabel('Epoch')
#     axs[1].legend(['train', 'val'], loc='best')

#     fig.savefig('plot.png')
#     plt.show()

# # Paths to training and validation datasets
# train_dir = r'D:\lama2\EmotionDetection_RealTime\data\data\train'  
# val_dir = r'D:\lama2\EmotionDetection_RealTime\data\data\test'

# # Hyperparameters
# num_train = 28709
# num_val = 7178
# batch_size = 64
# num_epoch = 50

# # Data augmentation
# train_datagen = ImageDataGenerator(rescale=1. / 255)
# val_datagen = ImageDataGenerator(rescale=1. / 255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(48, 48),
#     batch_size=batch_size,
#     color_mode="grayscale",
#     class_mode='categorical'
# )

# validation_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(48, 48),
#     batch_size=batch_size,
#     color_mode="grayscale",
#     class_mode='categorical'
# )

# # Model creation
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(7, activation='softmax'))

# # Training or displaying the model
# if mode == "train":
#     model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])
#     model_info = model.fit(
#         train_generator,
#         steps_per_epoch=num_train // batch_size,
#         epochs=num_epoch,
#         validation_data=validation_generator,
#         validation_steps=num_val // batch_size
#     )

#     plot_model_history(model_info)
#     model.save_weights('model.h5')

# elif mode == "display":
#     model.load_weights('model.h5')

#     # Define emotion dictionary for display
#     emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

#     # Start webcam feed
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
#             prediction = model.predict(cropped_img)
#             maxindex = int(np.argmax(prediction))
#             cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         cv2.imshow("Frame", frame)
#         key = cv2.waitKey(1) & 0xFF
        
#         if key == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
# else:
#     print("Invalid mode. Use 'train' or 'display'.")





import numpy as np
import argparse
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

# Suppress TensorFlow debugging logs and oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Command-line argument for mode selection
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display/test", required=True)
a = ap.parse_args()
mode = a.mode

def plot_model_history(model_history):
    """Plot training history for accuracy and loss."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')

    fig.savefig('plot.png')
    plt.show()

# Paths to training and validation datasets
train_dir = 'data/train'
val_dir = 'data/test'

# Hyperparameters
num_train = 28709
num_val = 7178
batch_size = 32
num_epoch = 10

# Ensure dataset directories exist
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    print(f"Error: Data directories '{train_dir}' or '{val_dir}' not found.")
    exit(1)

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical'
)

# Model creation
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Training, testing, or displaying the model
if mode == "train":
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size
    )

    plot_model_history(model_info)
    model.save_weights('model.h5')

elif mode == "test":
    model.load_weights('model.h5')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    loss, accuracy = model.evaluate(validation_generator, steps=num_val // batch_size)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

elif mode == "display":
    model.load_weights('model.h5')

    # Define emotion dictionary for display
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Start webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid mode. Use 'train', 'test', or 'display'.")
