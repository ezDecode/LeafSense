#!/usr/bin/env python3
"""
LeafSense Model Training Script
Trains a ResNet50-based model for crop leaf disease detection
"""

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
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LeafSenseTrainer:
    def __init__(self, data_dir='data', model_save_dir='saved_models'):
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 30
        self.learning_rate = 0.0001
        
        # Ensure directories exist
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Initialize data generators
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        
        # Model and history
        self.model = None
        self.history = None
        
    def setup_data_generators(self):
        """Setup data generators with augmentation for training and validation"""
        logger.info("Setting up data generators...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Only rescaling for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training generator
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        # Validation generator
        self.val_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'val'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Test generator
        self.test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Save class mapping
        self.class_indices = self.train_generator.class_indices
        self.num_classes = len(self.class_indices)
        
        logger.info(f"Found {self.num_classes} classes")
        logger.info(f"Training samples: {self.train_generator.samples}")
        logger.info(f"Validation samples: {self.val_generator.samples}")
        logger.info(f"Test samples: {self.test_generator.samples}")
        
        # Save class mapping
        with open(os.path.join(self.model_save_dir, 'class_indices.json'), 'w') as f:
            json.dump(self.class_indices, f, indent=2)
            
    def build_model(self):
        """Build the ResNet50-based model with custom classification head"""
        logger.info("Building model...")
        
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create the model
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model built successfully")
        self.model.summary()
        
    def train_model(self):
        """Train the model with callbacks"""
        logger.info("Starting model training...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_save_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        
    def fine_tune_model(self):
        """Fine-tune the model by unfreezing some layers"""
        logger.info("Starting fine-tuning...")
        
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[1]  # ResNet50 layer
        base_model.trainable = True
        
        # Freeze the bottom layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
            
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tune for a few more epochs
        fine_tune_epochs = 10
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_save_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            self.train_generator,
            epochs=fine_tune_epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Fine-tuning completed")
        
    def evaluate_model(self):
        """Evaluate the model on test data"""
        logger.info("Evaluating model...")
        
        # Load best model
        best_model_path = os.path.join(self.model_save_dir, 'best_model.h5')
        if os.path.exists(best_model_path):
            self.model = tf.keras.models.load_model(best_model_path)
            logger.info("Loaded best model for evaluation")
        
        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=1)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")
        
        # Predictions
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Classification report
        class_names = list(self.class_indices.keys())
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=class_names,
            output_dict=True
        )
        
        logger.info("Classification Report:")
        logger.info(classification_report(true_classes, predicted_classes, target_names=class_names))
        
        # Save detailed report
        with open(os.path.join(self.model_save_dir, 'evaluation_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
            
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        self.plot_confusion_matrix(cm, class_names)
        
        return test_accuracy, report
        
    def plot_confusion_matrix(self, cm, class_names):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(20, 16))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            logger.warning("No training history available")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_model_summary(self):
        """Save model summary to text file"""
        summary_path = os.path.join(self.model_save_dir, 'model_summary.txt')
        
        with open(summary_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            
        logger.info(f"Model summary saved to {summary_path}")
        
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            logger.info("Starting LeafSense training pipeline...")
            
            # Setup data generators
            self.setup_data_generators()
            
            # Build model
            self.build_model()
            
            # Train model
            self.train_model()
            
            # Fine-tune model
            self.fine_tune_model()
            
            # Evaluate model
            accuracy, report = self.evaluate_model()
            
            # Plot results
            self.plot_training_history()
            
            # Save model summary
            self.save_model_summary()
            
            logger.info("Training pipeline completed successfully!")
            logger.info(f"Final test accuracy: {accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            return False

def main():
    """Main function to run training"""
    # Check if data directory exists
    if not os.path.exists('data'):
        logger.error("Data directory not found. Please ensure you have the dataset organized in data/train, data/val, data/test directories.")
        return
        
    # Initialize trainer
    trainer = LeafSenseTrainer()
    
    # Run training pipeline
    success = trainer.run_training_pipeline()
    
    if success:
        logger.info("Training completed successfully! Check the saved_models directory for results.")
    else:
        logger.error("Training failed. Check the logs for details.")

if __name__ == "__main__":
    main()
