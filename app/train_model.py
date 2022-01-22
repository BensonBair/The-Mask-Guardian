from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import Sequential, models
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt

INPUT_PATH = '/Users/bensonbair/Documents/OOP_project/input'

# Load train and test set
train_dir = f"{INPUT_PATH}/face-mask-12k-images-dataset/Train"
test_dir = f"{INPUT_PATH}/face-mask-12k-images-dataset/Test"
val_dir = f"{INPUT_PATH}/face-mask-12k-images-dataset/Validation"

# Generate batches of tensor image data with real-time data augmentation


def get_train_generator():
    train_datagen = ImageDataGenerator(
        rescale=1.0/255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir, target_size=(128, 128), class_mode='categorical', batch_size=32)
    return train_generator


def get_val_generator():
    val_datagen = ImageDataGenerator(rescale=1.0/255)
    train_datagen = ImageDataGenerator(
        rescale=1.0/255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
    val_generator = val_datagen.flow_from_directory(directory=val_dir, target_size=(
        128, 128), class_mode='categorical', batch_size=32)
    return val_generator


def get_test_generator():
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    train_datagen = ImageDataGenerator(
        rescale=1.0/255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
    test_generator = test_datagen.flow_from_directory(
        directory=val_dir, target_size=(128, 128), class_mode='categorical', batch_size=32)
    return test_generator


def get_vgg_model():
    vgg19 = VGG19(weights='imagenet', include_top=False,
                  input_shape=(128, 128, 3))

    for layer in vgg19.layers:
        layer.trainable = False

    model = Sequential()
    model.add(vgg19)
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    print(model.summary())

    return model


def train(batch_size=32, epochs=20):
    train_generator = get_train_generator()
    val_generator = get_val_generator()
    test_generator = get_test_generator()
    model = get_vgg_model()

    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics="accuracy")
    history = model.fit_generator(generator=train_generator, steps_per_epoch=len(train_generator)//batch_size,
                                  epochs=epochs, validation_data=val_generator, validation_steps=len(val_generator)//batch_size)
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['validation'], loc='upper left')
    # plt.show()
    plt.savefig("model_accuracy.jpeg")
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation'], loc='upper left')
    # plt.show()
    plt.savefig("model_loss.jpeg")
    plt.close()
    print("metrics: ", model.evaluate_generator(test_generator))
    model.save(f"{INPUT_PATH}/finetuned_model/masknet2.h5")


def inference_demo(model):
    model = models.load_model(f"{INPUT_PATH}/finetuned_model/masknet2.h5")

    sample_mask_img = cv2.imread(
        f"{INPUT_PATH}/face-mask-12k-images-dataset/Test/WithMask/1565.png")
    sample_mask_img = cv2.resize(sample_mask_img, (128, 128))
    plt.imshow(sample_mask_img)
    # plt.show()
    sample_mask_img = np.reshape(sample_mask_img, [1, 128, 128, 3])
    sample_mask_img = sample_mask_img/255.0
    print('origin', model.predict(sample_mask_img))


def main():
    train()


if __name__ == '__main__':
    main()
