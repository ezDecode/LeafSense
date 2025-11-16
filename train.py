import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ==================== STEP 1: LOAD DATA ====================
def load_data(data_dir='data', img_size=160, batch_size=32):
    """Load images from directory and split into train/validation sets"""
    train = keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset='training', 
        seed=42, image_size=(img_size, img_size), batch_size=batch_size)
    
    val = keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset='validation',
        seed=42, image_size=(img_size, img_size), batch_size=batch_size)
    
    # Prefetch for performance
    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)
    
    return train, val, train.class_names


# ==================== STEP 2: BUILD MODEL ====================
def create_model(num_classes, img_size=160):
    """Build CNN using pretrained EfficientNetB4 (frozen backbone + custom head)"""
    base = keras.applications.EfficientNetB4(
        include_top=False, weights='imagenet',
        input_shape=(img_size, img_size, 3))
    base.trainable = False  # Freeze pretrained layers
    
    preprocess_input = keras.applications.efficientnet.preprocess_input
    
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
    return model


# ==================== STEP 3: TRAIN & SAVE ====================
def train_and_save(data_dir='data', epochs=10, output_dir='saved_models'):
    """Complete training pipeline: load → train → save"""
    # Load data
    print('Loading dataset...')
    train, val, classes = load_data(data_dir)
    print(f'Found {len(classes)} classes: {classes[:5]}...')
    
    # Build model
    print('\nBuilding model...')
    model = create_model(len(classes))
    model.summary()
    
    # Train
    print('\nTraining...')
    model.fit(train, validation_data=val, epochs=epochs)
    
    # Save model and class mapping
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save('plant_disease.keras')
    model.save(f'{output_dir}/best_model.keras')  # Backup copy
    
    with open(f'{output_dir}/class_indices.json', 'w') as f:
        json.dump({str(i): name for i, name in enumerate(classes)}, f, indent=2)
    
    print(f'\nModel saved to plant_disease.keras')
    print(f'Backup saved to {output_dir}/best_model.keras')
    
    # Quick evaluation
    print('Validation Accuracy:', model.evaluate(val, verbose=0)[1])
    return model


# ==================== RUN ====================
if __name__ == '__main__':
    train_and_save(
        data_dir='data',
        epochs=10,
        output_dir='saved_models'  
    )
