import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

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


def prepare_checkpoint_callback(
    checkpoint_path: str = "./checkpoints/best_performer.weights.h5",
    monitor: str = "val_accuracy",
    verbose: int = 1,
    save_weights_only: bool = True,
    save_best_only: bool = True,
    save_freq: str = "epoch"
) -> ModelCheckpoint:
    """
    Prepares and returns a ModelCheckpoint callback to save the best model weights during training.

    This function helps in saving the model's progress by storing the best-performing weights based 
    on a monitored metric (e.g., validation accuracy). The checkpoint can either save only the 
    weights or the entire model.

    Note:
        - If `save_weights_only=True`, the `checkpoint_path` **must end with `.weights.h5`** 
          to ensure only the weights are saved.
        - If `save_weights_only=False`, the full model (architecture + optimizer state) is saved.

    Args:
        checkpoint_path (str, optional): Path to save the checkpoint file. Defaults to "./checkpoints/best_performer.weights.h5".
        monitor (str, optional): Metric to monitor for improvements (e.g., "val_loss" or "val_accuracy"). Defaults to "val_accuracy".
        verbose (int, optional): Verbosity mode (0 = silent, 1 = updates on saving). Defaults to 1.
        save_weights_only (bool, optional): If True, only saves model weights. Defaults to True.
        save_best_only (bool, optional): If True, saves only the best model based on `monitor`. Defaults to True.
        save_freq (str, optional): Frequency at which checkpoints are saved ("epoch" or an integer for batch frequency). Defaults to "epoch".

    Returns:
        ModelCheckpoint: A configured ModelCheckpoint callback for model training.
    """
    return ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        verbose=verbose,
        save_weights_only=save_weights_only,
        save_best_only=save_best_only,
        save_freq=save_freq
    )


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
