import tensorflow as tf

def image_preprocessing(
    image: tf.Tensor, 
    label: tf.Tensor, 
    img_size: int = 224, 
    normalize: bool = False, 
    normalization_factor: float = 255.0
):
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