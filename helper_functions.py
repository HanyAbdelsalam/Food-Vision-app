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
