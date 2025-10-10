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

# GPU Configuration for RTX 3050 optimization
def configure_gpu():
    """Configure GPU settings for optimal performance on RTX 3050"""
    print("🔧 Configuring GPU for RTX 3050...")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✅ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
                
            # Set GPU as the default device
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            # Enable mixed precision for faster training on RTX 3050
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("✅ Mixed precision enabled (float16) for faster training")
            except Exception as e:
                print(f"⚠️  Mixed precision not available: {e}")
            
            return True
            
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
            return False
    else:
        print("❌ No GPU found! Training will use CPU (slower).")
        print("💡 Install CUDA Toolkit 11.2 and cuDNN 8.1 for GPU acceleration")
        print("   Run: install_cuda_rtx3050.bat for automatic installation")
        return False

def check_gpu_tensorflow():
    """Check TensorFlow GPU configuration"""
    print("\n🔍 TensorFlow GPU Information:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # Use newer API if available
    if hasattr(tf.config, 'list_physical_devices'):
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}")
        if gpus:
            print("✅ CUDA GPUs detected:")
            for gpu in gpus:
                print(f"   {gpu}")
                
            # Test GPU computation
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                    c = tf.matmul(a, b)
                print("✅ GPU computation test passed")
                return True
            except Exception as e:
                print(f"❌ GPU computation test failed: {e}")
                return False
        else:
            print("❌ No CUDA GPUs detected")
            return False
    else:
        # Fallback for older TensorFlow versions
        try:
            print(f"GPU available: {tf.test.is_gpu_available()}")
            return tf.test.is_gpu_available()
        except:
            print("GPU available: Unknown")
            return False

# Initialize GPU configuration
print("🚀 Initializing LeafSense GPU-Optimized Training System")
print("=" * 60)
GPU_AVAILABLE = configure_gpu()
check_gpu_tensorflow()
print("=" * 60)

class LeafSenseTrainer:
    def __init__(self):
        # Basic settings - OPTIMIZED FOR RTX 3050
        self.data_dir = 'data'
        self.model_save_dir = 'saved_models'
        self.img_size = (224, 224)
        
        # Optimized batch size for RTX 3050 (8GB VRAM)
        # Larger batch size for better GPU utilization
        self.batch_size = 64 if GPU_AVAILABLE else 32
        self.epochs = 50
        self.learning_rate = 0.0001
        
        # RTX 3050 optimizations
        self.use_mixed_precision = GPU_AVAILABLE
        self.prefetch_buffer = tf.data.AUTOTUNE if GPU_AVAILABLE else 1
        
        print(f"🚀 Trainer initialized for {'GPU (RTX 3050)' if GPU_AVAILABLE else 'CPU'}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🎯 Mixed precision: {'Enabled' if self.use_mixed_precision else 'Disabled'}")
        
        # Resume training settings
        self.checkpoint_dir = os.path.join(self.model_save_dir, 'checkpoints')
        self.training_state_file = os.path.join(self.model_save_dir, 'training_state.json')
        self.resume_from_epoch = 0
        self.initial_epoch = 0
        
        # Make sure directories exist
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
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
        print("Preparing data with improved augmentation and GPU optimization...")
        
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
        
        # Load training data with GPU optimization
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
        
        # GPU Optimization: Convert to tf.data for better performance
        if GPU_AVAILABLE:
            print("🚀 Applying GPU optimizations to data pipeline...")
            
            # Convert generators to tf.data datasets for better GPU utilization
            self.train_dataset = tf.data.Dataset.from_generator(
                lambda: self.train_generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.train_generator.num_classes), dtype=tf.float32)
                )
            ).prefetch(self.prefetch_buffer)
            
            self.val_dataset = tf.data.Dataset.from_generator(
                lambda: self.val_generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.val_generator.num_classes), dtype=tf.float32)
                )
            ).prefetch(self.prefetch_buffer)
        
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
        
        print(f"🚀 Optimized batch size for RTX 3050: {self.batch_size}")
        if GPU_AVAILABLE:
            print("✅ Data pipeline optimized for GPU training")
        
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
        print("Creating improved model with GPU optimizations...")
        
        # Check if model already exists
        if os.path.exists(f'{self.model_save_dir}/best_model.keras'):
            print("Loading existing model...")
            self.model = tf.keras.models.load_model(f'{self.model_save_dir}/best_model.keras')
            return
        
        # Use ResNet50 as base with GPU-optimized settings
        with tf.device('/GPU:0' if GPU_AVAILABLE else '/CPU:0'):
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            
            # Freeze base model initially
            base_model.trainable = False
            
            # IMPROVED architecture with BatchNormalization and mixed precision support
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D()(x)
            x = BatchNormalization()(x)  # Added
            x = Dense(512, activation='relu')(x)  # Increased size
            x = Dropout(0.5)(x)
            x = BatchNormalization()(x)  # Added
            x = Dense(256, activation='relu')(x)  # Added layer
            x = Dropout(0.3)(x)
            
            # Output layer - use float32 for mixed precision
            if self.use_mixed_precision:
                outputs = Dense(self.num_classes, activation='softmax', dtype='float32', name='predictions')(x)
            else:
                outputs = Dense(self.num_classes, activation='softmax')(x)
            
            self.model = Model(inputs, outputs)
        
        # Compile model with GPU-optimized settings
        optimizer = Adam(learning_rate=self.learning_rate)
        
        # For mixed precision, wrap optimizer
        if self.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        print("✅ GPU-optimized model created!")
        print(f"📊 Model has {self.model.count_params():,} parameters")
        print(f"🎯 Mixed precision: {'Enabled' if self.use_mixed_precision else 'Disabled'}")
        if GPU_AVAILABLE:
            print("🚀 Model will train on GPU for faster performance")
        
    def train_initial(self):
        print("Starting initial training with GPU optimization and resume support...")
        
        # Check if we should resume
        state = self.load_training_state()
        
        if state and state.get('phase') == 'initial' and self.resume_from_epoch > 0:
            print(f"Resuming initial training from epoch {self.resume_from_epoch}")
            
            # Load the latest checkpoint
            checkpoint_info = self.find_latest_checkpoint()
            if checkpoint_info:
                checkpoint_path, checkpoint_epoch = checkpoint_info
                print(f"Loading checkpoint from epoch {checkpoint_epoch}")
                self.model = tf.keras.models.load_model(checkpoint_path)
            
            # Adjust epochs
            remaining_epochs = self.epochs - self.resume_from_epoch
            if remaining_epochs <= 0:
                print("Initial training already completed, moving to fine-tuning...")
                return
        else:
            remaining_epochs = self.epochs
            self.initial_epoch = 0
        
        # GPU-optimized callbacks
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10,
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
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Add epoch checkpoint callback
        epoch_checkpoint = self.create_checkpoint_callback('initial')
        
        # Use GPU-optimized data if available
        train_data = getattr(self, 'train_dataset', self.train_generator)
        val_data = getattr(self, 'val_dataset', self.val_generator)
        
        print(f"🚀 Training on {'GPU' if GPU_AVAILABLE else 'CPU'} with batch size {self.batch_size}")
        
        # Train with resume support and GPU optimization
        self.history = self.model.fit(
            train_data,
            epochs=self.epochs,
            initial_epoch=self.initial_epoch,  # Resume from here
            validation_data=val_data,
            callbacks=[early_stop, model_checkpoint, reduce_lr, epoch_checkpoint],
            class_weight=self.class_weights,
            verbose=1
        )
        
        print("✅ Initial training done!")
        
    def fine_tune(self):
        print("Fine-tuning model with GPU optimization...")
        
        # Unfreeze more layers for fine-tuning
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Freeze early layers, unfreeze more layers than before
        for layer in base_model.layers[:-30]:  # Unfreeze last 30 layers instead of 20
            layer.trainable = False
            
        # Compile with GPU-optimized settings and lower learning rate
        optimizer = Adam(learning_rate=self.learning_rate/5)  # Even lower
        
        # For mixed precision, wrap optimizer
        if self.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            
        self.model.compile(
            optimizer=optimizer,
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
        
        # Use GPU-optimized data if available
        train_data = getattr(self, 'train_dataset', self.train_generator)
        val_data = getattr(self, 'val_dataset', self.val_generator)
        
        print(f"🔧 Fine-tuning on {'GPU' if GPU_AVAILABLE else 'CPU'}")
        
        fine_tune_history = self.model.fit(
            train_data,
            epochs=20,  # More fine-tuning epochs
            validation_data=val_data,
            callbacks=[early_stop, model_checkpoint, reduce_lr],
            class_weight=self.class_weights,  # Keep using class weights
            verbose=1
        )
        
        # Combine histories
        if hasattr(self, 'history') and self.history:
            for key in self.history.history.keys():
                if key in fine_tune_history.history:
                    self.history.history[key].extend(fine_tune_history.history[key])
        
        print("✅ Fine-tuning done!")
        
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
        
    def save_training_state(self, epoch, phase='initial'):
        """Save current training state"""
        state = {
            'last_completed_epoch': epoch,
            'phase': phase,  # 'initial' or 'fine_tune'
            'total_epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
        
        with open(self.training_state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Training state saved: epoch {epoch}, phase: {phase}")

    def load_training_state(self):
        """Load previous training state if exists"""
        if os.path.exists(self.training_state_file):
            with open(self.training_state_file, 'r') as f:
                state = json.load(f)
            
            self.resume_from_epoch = state.get('last_completed_epoch', 0)
            self.initial_epoch = self.resume_from_epoch
            
            print(f"Found previous training state:")
            print(f"  - Last completed epoch: {self.resume_from_epoch}")
            print(f"  - Phase: {state.get('phase', 'unknown')}")
            
            return state
        return None

    def find_latest_checkpoint(self):
        """Find the latest checkpoint file"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.keras')]
        if not checkpoint_files:
            return None
        
        # Extract epoch numbers and find the latest
        epochs = []
        for f in checkpoint_files:
            try:
                epoch_num = int(f.split('_')[2].split('.')[0])
                epochs.append((epoch_num, f))
            except:
                continue
        
        if epochs:
            latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
            return os.path.join(self.checkpoint_dir, latest_file), latest_epoch
        
        return None
    
    def create_checkpoint_callback(self, phase='initial'):
        """Create callback to save model after each epoch"""
        class EpochCheckpoint(tf.keras.callbacks.Callback):
            def __init__(self, checkpoint_dir, state_file, trainer, phase):
                super().__init__()
                self.checkpoint_dir = checkpoint_dir
                self.state_file = state_file
                self.trainer = trainer
                self.phase = phase
            
            def on_epoch_end(self, epoch, logs=None):
                # Save model checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.keras')
                self.model.save(checkpoint_path)
                
                # Save training state
                self.trainer.save_training_state(epoch + 1, self.phase)
                
                # Clean up old checkpoints (keep only last 3)
                self.cleanup_old_checkpoints()
            
            def cleanup_old_checkpoints(self):
                checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                                  if f.startswith('checkpoint_epoch_') and f.endswith('.keras')]
                if len(checkpoint_files) > 3:
                    # Sort by epoch number
                    epochs_files = []
                    for f in checkpoint_files:
                        try:
                            epoch_num = int(f.split('_')[2].split('.')[0])
                            epochs_files.append((epoch_num, f))
                        except:
                            continue
                    
                    epochs_files.sort(key=lambda x: x[0])
                    
                    # Remove oldest checkpoints, keep only last 3
                    for epoch_num, filename in epochs_files[:-3]:
                        try:
                            os.remove(os.path.join(self.checkpoint_dir, filename))
                            print(f"Cleaned up old checkpoint: {filename}")
                        except:
                            pass
        
        return EpochCheckpoint(self.checkpoint_dir, self.training_state_file, self, phase)
    
    def train_complete_model(self):
        print("Starting GPU-OPTIMIZED training process with RESUME support...")
        print("🎯 RTX 3050 Optimizations:")
        print(f"   - GPU Detection: {'✅ Enabled' if GPU_AVAILABLE else '❌ Not Available'}")
        print(f"   - Mixed Precision: {'✅ Enabled' if self.use_mixed_precision else '❌ Disabled'}")
        print(f"   - Optimized Batch Size: {self.batch_size}")
        print(f"   - Memory Growth: {'✅ Enabled' if GPU_AVAILABLE else '❌ N/A'}")
        
        # Check if we're resuming
        state = self.load_training_state()
        if state:
            print("🔄 RESUMING PREVIOUS TRAINING")
            print(f"Last completed epoch: {state.get('last_completed_epoch', 0)}")
            print(f"Phase: {state.get('phase', 'unknown')}")
        else:
            print("🆕 STARTING NEW TRAINING")
        
        print("Enhanced Features:")
        print("- GPU acceleration with CUDA/cuDNN")
        print("- Mixed precision training (float16)")
        print("- Optimized data pipeline")
        print("- Resume training from interruption")
        print("- Automatic checkpoint saving")
        print("- Class weights for imbalanced data")
        print("- Improved data augmentation")
        print("-" * 50)
        
        # Step 1: Prepare data
        self.prepare_data()
        
        # Step 2: Create or load model
        self.create_model()
        
        # Step 3: Check if we need to resume from checkpoint
        if state and self.resume_from_epoch > 0:
            checkpoint_info = self.find_latest_checkpoint()
            if checkpoint_info:
                checkpoint_path, checkpoint_epoch = checkpoint_info
                print(f"Loading model from checkpoint: epoch {checkpoint_epoch}")
                self.model = tf.keras.models.load_model(checkpoint_path)
        
        # Display GPU memory info if available
        if GPU_AVAILABLE:
            try:
                gpu_details = tf.config.experimental.get_device_details(tf.config.list_logical_devices('GPU')[0])
                print(f"🚀 GPU Details: {gpu_details}")
            except:
                print("🚀 GPU training enabled")
        
        # Step 4: Initial training (with resume)
        if not state or state.get('phase') == 'initial':
            self.train_initial()
        
        # Step 5: Fine-tune (with resume)
        if not state or state.get('phase') in ['initial', 'fine_tune']:
            self.fine_tune()
        
        # Step 6: Test model
        accuracy = self.test_model()
        
        # Step 7: Plot results
        self.plot_training_curves()
        
        # Clean up training state (training completed)
        if os.path.exists(self.training_state_file):
            os.remove(self.training_state_file)
            print("Training completed successfully - state file cleaned up")
        
        print(f"🎉 Training complete! Final accuracy: {accuracy:.4f}")
        print(f"🚀 Trained using: {'GPU (RTX 3050)' if GPU_AVAILABLE else 'CPU'}")
        return accuracy

# Main function
def main():
    print("=" * 60)
    print("🌿 LeafSense Plant Disease Classification Training")
    print("🚀 GPU-Optimized for RTX 3050")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists('data'):
        print("❌ Error: data folder not found!")
        print("Make sure you have data/train, data/val, and data/test folders")
        return
        
    print("✅ Data folder found!")
    
    # Create trainer and start training
    trainer = LeafSenseTrainer()
    
    # Start training with all optimizations
    trainer.train_complete_model()
    
    print("\n" + "=" * 60)
    print("🎉 Training completed!")
    print("📁 Check the saved_models folder for results:")
    print("   - best_model.keras (trained model)")
    print("   - improved_test_results.json (detailed results)")
    print("   - improved_training_curves.png (training plots)")
    print("   - improved_confusion_matrix.png (confusion matrix)")
    print("   - class_indices.json (class mappings)")
    if GPU_AVAILABLE:
        print("🚀 Training was accelerated using GPU!")
    else:
        print("⚠️  Training used CPU - consider installing CUDA for faster training")
    print("=" * 60)

if __name__ == "__main__":
    main()
