#!/usr/bin/env python3
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class LeafSenseTrainer:
    def __init__(self):
        # Basic settings
        self.data_dir = 'data'
        self.model_save_dir = 'saved_models'
        self.img_size = (224, 224)
        self.batch_size = 64  # Increased batch size
        self.epochs = 25
        self.learning_rate = 0.001
        
        # Make sure directories exist
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        # Variables to store stuff
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.model = None
        self.history = None
        self.class_indices = None
        self.num_classes = 0
        
    def prepare_data(self):
        print("Preparing data...")
        
        # Training data with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Validation and test data - just rescale
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        # Load validation data
        self.val_generator = val_test_datagen.flow_from_directory(
            'data/val',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Load test data
        self.test_generator = val_test_datagen.flow_from_directory(
            'data/test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Get class info
        self.class_indices = self.train_generator.class_indices
        self.num_classes = len(self.class_indices)
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Training images: {self.train_generator.samples}")
        print(f"Validation images: {self.val_generator.samples}")
        print(f"Test images: {self.test_generator.samples}")
        
        # Save class mapping
        with open(f'{self.model_save_dir}/class_indices.json', 'w') as f:
            json.dump(self.class_indices, f)
            
    def create_model(self):
        print("Creating model...")
        
        # Check if model already exists
        if os.path.exists(f'{self.model_save_dir}/best_model.h5'):
            print("Loading existing model...")
            self.model = tf.keras.models.load_model(f'{self.model_save_dir}/best_model.h5')
            return
        
        # Use ResNet50 as base
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze base model
        base_model.trainable = False
        
        # Add our own layers on top
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model created!")
        
    def train_initial(self):
        print("Starting training...")
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        
        model_checkpoint = ModelCheckpoint(
            f'{self.model_save_dir}/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
        
        # Train
        self.history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            callbacks=[early_stop, model_checkpoint, reduce_lr]
        )
        
        print("Initial training done!")
        
    def fine_tune(self):
        print("Fine-tuning model...")
        
        # Unfreeze some layers
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Freeze early layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
            
        # Compile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate/10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train again
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model_checkpoint = ModelCheckpoint(
            f'{self.model_save_dir}/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
        
        fine_tune_history = self.model.fit(
            self.train_generator,
            epochs=10,
            validation_data=self.val_generator,
            callbacks=[early_stop, model_checkpoint]
        )
        
        print("Fine-tuning done!")
        
    def test_model(self):
        print("Testing model...")
        
        # Load best model
        self.model = tf.keras.models.load_model(f'{self.model_save_dir}/best_model.h5')
        
        # Test
        test_loss, test_accuracy = self.model.evaluate(self.test_generator)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Get predictions
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Classification report
        class_names = list(self.class_indices.keys())
        report = classification_report(true_classes, predicted_classes, target_names=class_names)
        print("Classification Report:")
        print(report)
        
        # Save report
        report_dict = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)
        with open(f'{self.model_save_dir}/test_results.json', 'w') as f:
            json.dump(report_dict, f, indent=2)
            
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        self.plot_confusion_matrix(cm, class_names)
        
        return test_accuracy
        
    def plot_confusion_matrix(self, cm, class_names):
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.model_save_dir}/confusion_matrix.png')
        plt.close()
        
    def plot_training_curves(self):
        if self.history is None:
            return
            
        plt.figure(figsize=(12, 4))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.model_save_dir}/training_curves.png')
        plt.close()
        
    def train_complete_model(self):
        print("Starting complete training process...")
        
        # Step 1: Prepare data
        self.prepare_data()
        
        # Step 2: Create model
        self.create_model()
        
        # Step 3: Initial training
        self.train_initial()
        
        # Step 4: Fine-tune
        self.fine_tune()
        
        # Step 5: Test model
        accuracy = self.test_model()
        
        # Step 6: Plot results
        self.plot_training_curves()
        
        print(f"Training complete! Final accuracy: {accuracy:.4f}")
        return accuracy

# Main function
def main():
    # Check if data exists
    if not os.path.exists('data'):
        print("Error: data folder not found!")
        print("Make sure you have data/train, data/val, and data/test folders")
        return
        
    # Create trainer and start training
    trainer = LeafSenseTrainer()
    trainer.train_complete_model()
    
    print("All done! Check the saved_models folder for results.")

if __name__ == "__main__":
    main()
