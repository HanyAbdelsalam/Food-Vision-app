from typing import Type
import matplotlib.pyplot as plt
import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Activation, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import RandomCrop, RandomFlip, RandomHeight, RandomWidth, RandomZoom, RandomRotation, Rescaling
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import mixed_precision


def image_preprocessing(
    image: tf.Tensor, 
    label: tf.Tensor, 
    img_size: int = 224, 
    normalize: bool = False, 
    normalization_factor: float = 255.0
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Preprocesses an image by resizing, converting it to float32, 
    and optionally normalizing its pixel values.

    Args:
        image (tf.Tensor): Input image tensor.
        label (tf.Tensor): Corresponding label tensor.
        img_size (int, optional): Target image size for resizing (height, width). Defaults to 224.
        normalize (bool, optional): Whether to normalize pixel values. Defaults to False.
        normalization_factor (float, optional): Value to divide pixel values by if normalization is applied. Defaults to 255.0.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: A tuple containing the preprocessed image and label.
    """
    # Resize the image to the specified dimensions
    image = tf.image.resize(image, size=[img_size, img_size])
    
    # Convert image dtype to float32
    image = tf.cast(image, tf.float32)
    
    # Normalize pixel values if specified
    if normalize:
        image = image / normalization_factor
        
    return image, label


def prepare_callbacks(
    checkpoint_name: str,
    logs_name: str,
    monitor: str = "val_accuracy",
    verbose: int = 1,
    checkpoint_save_best_only: bool = True,
    checkpoint_save_weights_only: bool = True,
    checkpoint_save_freq: str = "epoch",
    reduceLearningRate_factor: float = 0.1,
    reduceLearningRate_patience: int = 3,
    earlyStopping_patience: int = 3
) -> list:
    """
    Prepares and returns a list of Keras callbacks for model training.

    Args:
        checkpoint_name (str): Base name for checkpoint files.
        logs_name (str): Name for the TensorBoard logs directory.
        monitor (str): Metric to monitor (default is "val_accuracy").
        verbose (int): Verbosity level for callbacks (0 or 1).
        checkpoint_save_best_only (bool): Save only the best model.
        checkpoint_save_weights_only (bool): Save only model weights.
        checkpoint_save_freq (str): Frequency for saving checkpoints ("epoch" or batch).
        reduceLearningRate_factor (float): Factor by which learning rate is reduced.
        reduceLearningRate_patience (int): Patience for reducing learning rate.
        earlyStopping_patience (int): Patience for early stopping.

    Returns:
        list: List of callbacks (ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping).
    """

    if checkpoint_save_weights_only:
        checkpoint_path = f"./checkpoints/{checkpoint_name}.weights.h5"
    else:
        checkpoint_path = f"./checkpoints/{checkpoint_name}.model.keras"

    logs_path = f"./logs/{logs_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # ModelCheckpoint callback: saves the model (or weights) based on the monitored metric.
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 monitor=monitor,
                                 save_best_only=checkpoint_save_best_only,
                                 save_weights_only=checkpoint_save_weights_only,
                                 save_freq=checkpoint_save_freq,
                                 verbose=verbose)
    
    # TensorBoard callback: logs training progress for visualization using TensorBoard.
    tensorboard = TensorBoard(log_dir=logs_path)
    
    # ReduceLROnPlateau callback: reduces the learning rate when the monitored metric stops improving.
    reduce_lr = ReduceLROnPlateau(monitor=monitor,
                                  factor=reduceLearningRate_factor,
                                  patience=reduceLearningRate_patience,
                                  verbose=verbose)
    
    # EarlyStopping callback: stops training when the monitored metric stops improving for a given number of epochs.
    early_stopping = EarlyStopping(monitor=monitor,
                                   patience=earlyStopping_patience,
                                   verbose=verbose)
    
    # Return the list of callbacks to be used during model training.
    return [checkpoint, tensorboard, reduce_lr, early_stopping]


def plot_history(history: tf.keras.callbacks.History):
    """
    Plots the training and validation accuracy and loss curves from a Keras History object.

    This function visualizes the model's learning progress by plotting accuracy and loss 
    trends over epochs for both training and validation.

    **Note:** 
        This function requires the History object to contain validation metrics (`val_accuracy` and `val_loss`). 
        It should only be used when the model was trained with validation data.

    Args:
        history (tf.keras.callbacks.History): Keras History object containing training and validation metrics.

    Raises:
        ValueError: If the history object does not contain validation accuracy (`val_accuracy`) or validation loss (`val_loss`).

    Example:
        >>> model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
        >>> plot_history(model.history)
    """
    # Ensure validation metrics exist
    if 'val_accuracy' not in history.history or 'val_loss' not in history.history:
        raise ValueError("The history object must contain 'val_accuracy' and 'val_loss'. Ensure the model was trained with validation data.")

    # Extract history data (keeping your original extraction style)
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy Plot
    axes[0].plot(accuracy, label='Training Accuracy', color='blue', marker='o')
    axes[0].plot(val_accuracy, label='Validation Accuracy', color='red', marker='o')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss Plot
    axes[1].plot(loss, label='Training Loss', color='blue', marker='o')
    axes[1].plot(val_loss, label='Validation Loss', color='red', marker='o')
    axes[1].set_title('Training and Validation Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.show()


def build_model(
    model_name: str,
    num_classes: int = 101,
    with_data_aug: bool = False,
    backbone: Type[Model] = EfficientNetB0,
    normalize: bool = False,
    add_bottleneck_layer: bool = False,
    with_regularization_and_dropout: bool = False,
    use_transfer_learning: bool = False) -> Model:
    """
    Builds a customizable image classification model.
    
    Parameters:
        model_name (str): Name of the model.
        num_classes (int): Number of output classes for the final classification layer.
        with_data_aug (bool): Whether to include data augmentation layers.
        backbone (Type[Model]): Pretrained CNN backbone model (e.g., EfficientNetB0) from `tf.keras.applications`.
        normalize (bool): Whether to normalize input images if data augmentation is enabled.
        add_bottleneck_layer (bool): Whether to add an extra dense bottleneck layer before classification.
        with_regularization_and_dropout (bool): Whether to apply dropout and L2 regularization.
        use_transfer_learning (bool): If True, freezes the backbone; otherwise, trains from scratch.
    
    Returns:
        Model: A compiled Keras model ready for training.
    
    Note:
        The `backbone` parameter must be a model from `tf.keras.applications` (e.g., EfficientNetB0, ResNet50, etc.)
    """
   
    mixed_precision.set_global_policy("mixed_float16")  # Set mixed precision policy
    print(mixed_precision.global_policy())  # Print global policy
    
    # Input layer for 224x224 RGB images
    inputs = Input(shape=(224, 224, 3), name="input_layer")
    
    # Load the backbone model without the top classification layers
    base_model = backbone(include_top=False, input_shape=(224, 224, 3))
    
    # Apply data augmentation if enabled
    if with_data_aug:
        if normalize:
            data_augmentation = Sequential([
                RandomFlip("horizontal"),
                RandomRotation(0.2),
                RandomHeight(0.2),
                RandomWidth(0.2),
                RandomZoom(0.2),
                Rescaling(1./255)  # Normalize images to [0,1] range
            ], name="data_augmentation_layer")
        else:
            data_augmentation = Sequential([
                RandomFlip("horizontal"),
                RandomRotation(0.2),
                RandomHeight(0.2),
                RandomWidth(0.2),
                RandomZoom(0.2)
            ], name="data_augmentation_layer")        
        x = data_augmentation(inputs)
    else:
        x = inputs
    
    # Freeze the backbone for transfer learning, otherwise train it
    base_model.trainable = not use_transfer_learning
    x = base_model(x, training=not use_transfer_learning)
    
    # Apply global average pooling to reduce feature map size
    x = GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    
    # Optional bottleneck layer to add extra dense features
    if add_bottleneck_layer:
        if with_regularization_and_dropout:
            x = Dropout(0.3)(x)  # Apply dropout to reduce overfitting
            x = Dense(256, activation="relu", name="bottleneck_layer", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        else:
            x = Dense(256, activation="relu", name="bottleneck_layer")(x)
    
    # Final classification layer
    if with_regularization_and_dropout:
        x = Dropout(0.3)(x)  # Apply dropout before logits layer
        x = Dense(num_classes, name="logits_layer", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    else:
        x = Dense(num_classes, name="logits_layer")(x)
    
    # Softmax activation for multi-class classification (101 classes)
    outputs = Activation("softmax", dtype=tf.float32, name="activation_output_layer")(x)
    
    # Create the model
    model = Model(inputs, outputs, name=model_name)
    
    # Compile the model with sparse categorical cross-entropy and Adam optimizer
    model.compile(loss=SparseCategoricalCrossentropy(),
                  optimizer=Adam(),
                  metrics=["accuracy"])
    
    return model