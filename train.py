#!/usr/bin/env python3
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class LeafSenseTrainer:
    def __init__(self):
        # Basic settings - IMPROVED PARAMETERS
        self.data_dir = 'data'
        self.model_save_dir = 'saved_models'
        self.img_size = (224, 224)
        self.batch_size = 32  # Reduced for better gradient updates
        self.epochs = 50      # Increased from 25
        self.learning_rate = 0.0001  # Reduced from 0.001
        
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
        self.class_weights = None
        
    def prepare_data(self):
        print("Preparing data with improved augmentation...")
        
        # IMPROVED data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,      # Increased
            width_shift_range=0.3,  # Increased  
            height_shift_range=0.3, # Increased
            horizontal_flip=True,
            zoom_range=0.3,         # Increased
            shear_range=0.2,        # Added
            brightness_range=[0.8, 1.2],  # Added
            fill_mode='nearest'
        )
        
        # Validation and test data - just rescale
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        # Load validation data
        self.val_generator = val_test_datagen.flow_from_directory(
            'data/val',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Load test data if it exists
        if os.path.exists('data/test') and os.listdir('data/test'):
            self.test_generator = val_test_datagen.flow_from_directory(
                'data/test',
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
        else:
            print("No test data found, will use validation data for final testing")
            self.test_generator = None
        
        # Get class info
        self.class_indices = self.train_generator.class_indices
        self.num_classes = len(self.class_indices)
        
        # CALCULATE CLASS WEIGHTS for imbalanced data
        self.calculate_class_weights()
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Training images: {self.train_generator.samples}")
        print(f"Validation images: {self.val_generator.samples}")
        if self.test_generator:
            print(f"Test images: {self.test_generator.samples}")
        else:
            print("Test images: 0 (will use validation set for testing)")
        
        # Save class mapping
        with open(f'{self.model_save_dir}/class_indices.json', 'w') as f:
            json.dump(self.class_indices, f)
    
    def calculate_class_weights(self):
        """Calculate class weights to handle imbalanced data"""
        print("Calculating class weights for imbalanced data...")
        
        # Get class counts
        class_counts = []
        for class_name in self.train_generator.class_indices.keys():
            class_dir = os.path.join('data/train', class_name)
            count = len(os.listdir(class_dir))
            class_counts.append(count)
        
        # Calculate weights
        total_samples = sum(class_counts)
        class_weights = {}
        
        for i, count in enumerate(class_counts):
            # Higher weight for classes with fewer samples
            weight = total_samples / (self.num_classes * count)
            class_weights[i] = weight
            
        self.class_weights = class_weights
        print(f"Class weights calculated. Range: {min(class_weights.values()):.2f} - {max(class_weights.values()):.2f}")
            
    def create_model(self):
        print("Creating improved model...")
        
        # Check if model already exists
        if os.path.exists(f'{self.model_save_dir}/best_model.keras'):
            print("Loading existing model...")
            self.model = tf.keras.models.load_model(f'{self.model_save_dir}/best_model.keras')
            return
        
        # Use ResNet50 as base
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze base model initially
        base_model.trainable = False
        
        # IMPROVED architecture with BatchNormalization
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)  # Added
        x = Dense(512, activation='relu')(x)  # Increased size
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)  # Added
        x = Dense(256, activation='relu')(x)  # Added layer
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile model with LOWER learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        print("Improved model created!")
        print(f"Model has {self.model.count_params():,} parameters")
        
    def train_initial(self):
        print("Starting initial training with class weights...")
        
        # IMPROVED callbacks
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10,  # Increased patience
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            f'{self.model_save_dir}/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5,  # More conservative reduction
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train with CLASS WEIGHTS
        self.history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            callbacks=[early_stop, model_checkpoint, reduce_lr],
            class_weight=self.class_weights,  # ADDED CLASS WEIGHTS
            verbose=1
        )
        
        print("Initial training done!")
        
    def fine_tune(self):
        print("Fine-tuning model...")
        
        # Unfreeze more layers for fine-tuning
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Freeze early layers, unfreeze more layers than before
        for layer in base_model.layers[:-30]:  # Unfreeze last 30 layers instead of 20
            layer.trainable = False
            
        # Compile with EVEN LOWER learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate/5),  # Even lower
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        # Train again with more patience
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=8, 
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            f'{self.model_save_dir}/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3,
            min_lr=1e-8,
            verbose=1
        )
        
        fine_tune_history = self.model.fit(
            self.train_generator,
            epochs=20,  # More fine-tuning epochs
            validation_data=self.val_generator,
            callbacks=[early_stop, model_checkpoint, reduce_lr],
            class_weight=self.class_weights,  # Keep using class weights
            verbose=1
        )
        
        # Combine histories
        if hasattr(self, 'history') and self.history:
            for key in self.history.history.keys():
                if key in fine_tune_history.history:
                    self.history.history[key].extend(fine_tune_history.history[key])
        
        print("Fine-tuning done!")
        
    def test_model(self):
        print("Testing model...")
        
        # Load best model
        self.model = tf.keras.models.load_model(f'{self.model_save_dir}/best_model.keras')
        
        # Use validation data if no test data available
        test_data = self.test_generator if self.test_generator else self.val_generator
        
        # Test
        test_results = self.model.evaluate(test_data, verbose=1)
        test_loss = test_results[0]
        test_accuracy = test_results[1]
        test_top3_accuracy = test_results[2] if len(test_results) > 2 else None
        
        print(f"Test accuracy: {test_accuracy:.4f}")
        if test_top3_accuracy:
            print(f"Test top-3 accuracy: {test_top3_accuracy:.4f}")
        
        # Get predictions
        test_data.reset()
        predictions = self.model.predict(test_data, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_data.classes
        
        # Classification report
        class_names = list(self.class_indices.keys())
        report = classification_report(true_classes, predicted_classes, target_names=class_names)
        print("Classification Report:")
        print(report)
        
        # Save results
        results = {
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "test_top3_accuracy": float(test_top3_accuracy) if test_top3_accuracy else None,
            "classification_report": classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True),
            "used_validation_for_testing": self.test_generator is None
        }
        
        with open(f'{self.model_save_dir}/improved_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        self.plot_confusion_matrix(cm, class_names)
        
        return test_accuracy
        
    def plot_confusion_matrix(self, cm, class_names):
        plt.figure(figsize=(20, 16))  # Larger figure for many classes
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Improved Model - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.model_save_dir}/improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_training_curves(self):
        if self.history is None:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Accuracy plot
        plt.subplot(2, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Loss plot
        plt.subplot(2, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Top-3 accuracy if available
        if 'top_3_accuracy' in self.history.history:
            plt.subplot(2, 2, 3)
            plt.plot(self.history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
            plt.plot(self.history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
            plt.title('Top-3 Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Top-3 Accuracy')
            plt.legend()
            plt.grid(True)
        
        # Learning rate plot
        if 'lr' in self.history.history:
            plt.subplot(2, 2, 4)
            plt.plot(self.history.history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_save_dir}/improved_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def train_complete_model(self):
        print("Starting IMPROVED training process...")
        print("Changes from original:")
        print("- Lower learning rate (0.0001 vs 0.001)")
        print("- More epochs (50 vs 25)")
        print("- Class weights for imbalanced data")
        print("- Improved data augmentation")
        print("- Better model architecture with BatchNorm")
        print("- Top-3 accuracy tracking")
        print("- More conservative learning rate reduction")
        print("-" * 50)
        
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
        
        print(f"Improved training complete! Final accuracy: {accuracy:.4f}")
        print("Check saved_models/ for results and plots")
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
    
    print("All done! Check the saved_models folder for improved results.")

if __name__ == "__main__":
    main()
