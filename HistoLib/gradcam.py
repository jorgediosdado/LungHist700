from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib as mpl
from tensorflow.keras.models import Model
import cv2


# The following two functions were copied from: 
# https://keras.io/examples/vision/grad_cam/

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
        
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)/255
    return array
        
        
def merge_image_mask(im, mk, alpha=0.3, colormap='jet', mask_thresh=0.5):
    """
    Given an image and a mask of the same size, it blends the images together.
    """
    
    # Ensure the mask doesn't have values out of [0-1] range.
    mk = np.clip(mk, 0, 1)

    heatmap = np.uint8(255 * mk)
    jet = mpl.colormaps[colormap]

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Blend the images together
    merge_image = jet_heatmap * alpha + im * (1-alpha)
    merge_image[mk<mask_thresh] = im[mk<mask_thresh]
    
    return merge_image
    
        
def generate_gradcam_samples(model, generator, N=8, 
                             mask_thresh=0.25, layer='conv5_block3_3_conv'):
    """
    Show some samples of the test generator. 
    """
    fig, axs = plt.subplots(3,N, figsize=(25,8))
    used = set()   
    for i in range(N):    
        
        # Ensure they are unique
        idx = np.random.randint(0, len(generator.images))
        while idx in used:
            idx = np.random.randint(0, len(generator.images))
        used.add(idx)
        
        x,y,z = generator[0][0][0].shape
        imsize = x,y
        
        im = get_img_array(generator.images[idx], imsize)
        
        # Original image
        axs[0,i].imshow(im[0]);
        axs[0,i].axis('off')
        axs[0,i].set_title(f"Real: {generator.images[idx].split('/')[2]} {generator.labels[idx]}", fontsize=18)
        
        heat = make_gradcam_heatmap(im, model, layer)
        pos_i, pos_j = imsize
        heat = cv2.resize(heat, dsize=(pos_j, pos_i))
        
        # Heatmap
        axs[1,i].imshow(heat)
        axs[1,i].axis('off')
        axs[1,i].set_title(f"Pred: {np.argmax(model.predict(im)[0])}", fontsize=18)
        
        # Merged
        axs[2,i].imshow(merge_image_mask(im[0], heat, mask_thresh=mask_thresh))
        axs[2,i].axis('off')


    plt.tight_layout()
    plt.show()