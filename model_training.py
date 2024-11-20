import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class Config:
    IMAGE_SIZE = (48, 48)
    BATCH_SIZE = 32  
    EPOCHS = 50
    NUM_CLASSES = 7
    DROPOUT_RATE = 0.5
    EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(image):

    image = tf.cast(image, tf.float32)
    
    image = image / 255.0
    return image

def load_and_preprocess_data(dataset_path):

    def process_directory(directory):
        images = []
        labels = []
        
        for emotion_idx, emotion_name in enumerate(Config.EMOTION_LABELS):
            emotion_path = os.path.join(directory, emotion_name)
            
            if not os.path.exists(emotion_path):
                print(f"Warning: {emotion_path} does not exist!")
                continue
            
            for img_name in os.listdir(emotion_path):
                img_path = os.path.join(emotion_path, img_name)
                
                img = tf.keras.preprocessing.image.load_img(
                    img_path, 
                    target_size=Config.IMAGE_SIZE, 
                    color_mode='grayscale'
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                
                images.append(img_array)
                labels.append(emotion_idx)
        
        return np.array(images), np.array(labels)
    
    
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    
    X_train, y_train = process_directory(train_dir)
    X_test, y_test = process_directory(test_dir)
   
    X_train = np.array([preprocess_image(img) for img in X_train])
    X_test = np.array([preprocess_image(img) for img in X_test])
    
    y_train = to_categorical(y_train, num_classes=Config.NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=Config.NUM_CLASSES)
    
    return X_train, X_test, y_train, y_test

def create_parallel_cnn():

    input_layer = layers.Input(shape=(*Config.IMAGE_SIZE, 1))

    
    def shallow_cnn(x):
        x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.3)(x)

        return x


    def medium_cnn(x):
        x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(128, (5,5), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.4)(x)

        return x


    def deep_cnn(x):
        x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(256, (5,5), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.5)(x)

        return x


    shallow_features = shallow_cnn(input_layer)
    medium_features = medium_cnn(input_layer)
    deep_features = deep_cnn(input_layer)


    shallow_flat = layers.GlobalAveragePooling2D()(shallow_features)
    medium_flat = layers.GlobalAveragePooling2D()(medium_features)
    deep_flat = layers.GlobalAveragePooling2D()(deep_features)

    merged_features = layers.concatenate([shallow_flat, medium_flat, deep_flat])


    x = layers.Dense(512, activation='relu')(merged_features)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)


    output_layer = layers.Dense(Config.NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(X_train, X_test, y_train, y_test):

    total_train_samples = len(X_train)
    steps_per_epoch = total_train_samples // Config.BATCH_SIZE
    
    print(f"Training samples: {total_train_samples}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    

    def augment(image, label):
       
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        return image, label
    
   
    train_dataset = (train_dataset
        .shuffle(buffer_size=total_train_samples)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(Config.BATCH_SIZE)
        .repeat() 
        .prefetch(tf.data.AUTOTUNE))
    
   
    val_dataset = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(Config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE))
    
   
    model = create_parallel_cnn()
    

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=steps_per_epoch * 5,  
        decay_rate=0.9,
        staircase=True
    )
    
  
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    

    model.summary()
    
   
    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            'tuned5.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    try:
     
        history = model.fit(
            train_dataset,
            epochs=Config.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    except Exception as e:
        print(f"Training error occurred: {str(e)}")
        raise


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test, y_test: Test data and labels
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

def main():
    tf.random.set_seed(42)
    
    try:

        dataset_path = 'dataset'
        X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)
        

        model, history = train_model(X_train, X_test, y_train, y_test)
        
      
        if model is not None:

            evaluate_model(model, X_test, y_test)
            

            model.save('balanced_facial_emotion_model.h5')
        else:
            print("Model training failed!")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()