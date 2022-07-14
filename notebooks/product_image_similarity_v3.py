# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] pycharm={"name": "#%% md\n"}
# # Product Image Similarity
# In this notebook, we will develop a model that takes an image as input, and finds closely related product images based on that. "Closely related," in this context, means images that "look" similar to the input image.
#
# ##### Overview of the process:
# 1. We will create a generator to hold the file paths of all the images in our dataset. We will then use this generator to explore our dataset, and later, feed it into our model.
# 2. We won't be "training" a new model; we'll just use an existing convolutional network (ResNet) and load its pre-trained weights. What we will be doing, however, is feeding the images in our dataset to get the images' embedding vectors.
# 3. When a user uploads an image and tries to find the most similar product image(s) from our dataset, we will need a mechanism to compare the uploaded image with all/most of the images in our dataset. The naive solution is to create a large table with the embeddings of every image and then iterating through the entire list. However, because websites have lots of product images, this is not scalable. Instead, we will use locality sensitive hashing to find the k-approximate nearest neighbors -- i.e. "similar" images. This will return relatively good results (but not the absolute best/closest matches) in an efficient (fast) manner.
# 4. While we are calculating the images' embedding vectors, we will also populate a lookup dict that references the image ID with the location of the image.
# 5. Finally, we will create a prediction function that returns a list of similar images for the given input. We'll then test our model on a few example product images to ensure it's working as expected.

# + pycharm={"name": "#%%\n"}
# Import libraries:

# Standard lib:
import glob

# ML/data science libraries:
import numpy as np
from annoy import AnnoyIndex

# TenserFlow/Keras classes:
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalMaxPool2D

# Plotting and imaging:
import cv2
import matplotlib.pyplot as plt


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Load Images:

# + pycharm={"name": "#%%\n"}
# N.b. we don’t need to shuffle the data since we’re not doing any training:
def get_file_paths():
    """ Create generator of all image file paths.

    :return {generator}
    """

    # We could also have used `pathlib.Path().glob()`, but that returns a POSIX path rather than str:
    return glob.iglob('../data/e-commerce-product-images/**/*.jpg', recursive=True)

file_paths = get_file_paths()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Exploring the Data:
#
# Let's look at the typical images in our dataset.
# N.b. the dataset includes different size images and images with different number of channels (i.e. both RGB and grayscale) images.

# + pycharm={"name": "#%%\n"}
example_img = cv2.imread(next(file_paths))
print(plt.imshow(example_img))

# reset generator:
file_paths = get_file_paths()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Load ResNet Model:

# + pycharm={"name": "#%%\n"}
# Constants:
# The model expects these image dimensions:
IMG_HEIGHT  = 224
IMG_WIDTH   = 224

resnet_model = ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
resnet_model.trainable = False

resnet_model.summary()

# + pycharm={"name": "#%%\n"}
# Update the model by adding a global max pooling layer:
# In essence, we replaced the last couple layers of the original ResNet model with a layer that outputs the embedding/features of the image.
# Recall that global pooling always reduces the output to be shape 1 x 1 x channels; essentially, outputting a layer of feature-maps.
model = Sequential([
    resnet_model,
    GlobalMaxPool2D()
])

model.summary()

# + pycharm={"name": "#%%\n"}
# Save model:

# N.b. we need to specify model version as the file directory, otherwise TensorFlow Serving will
# throw an error that there aren't any servable model versions: <https://stackoverflow.com/q/45544928>:
MODEL_VERSION = 3
model.save(f'../models/resnet_similarity/{MODEL_VERSION}')


# + [markdown] pycharm={"name": "#%% md\n"}
# **Aside:** if we didn't specify `include_top=False` in `resnet_model` from the previous two cells,
# we could alternatively have created a new model and get the last 3rd to last layer:
# ```
# resnet_model_ = ResNet50V2(weights='imagenet')
# ```
#
# This would, however, result in the exact same model:
# ```
# feature_extraction_model = Model(
#     name='ResNet50V2_ExtractFeature',
#     inputs=resnet_model_.inputs,
#     outputs=resnet_model_.get_layer('post_relu').output
# )
# ```

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Helper methods:

# + pycharm={"name": "#%%\n"}
def process_image(img: np.ndarray):
    """ Pre-process images before feeding to model.

    Resizes image, scales (/255), and expands array dimension. The model requires specific input dimensions (shape),
    therefore resizing and adding dimension is necessary. Scaling improves performance.
    TODO: ideally, we should use standard deviation and mean of dataset instead of simply dividing by 255.

    :param img:
    :return {np.ndarray}: returns processed image
    """

    # When serving (via TensorFlow Serving), the model signature expects float type as input (enter `saved_model_cli show`
    # in command line for details), but since cv2 reads images as float64, we need to convert it to float32. If we didn't cast the array,
    # we would get this error: `Expects arg[0] to be float but double is provided`. N.b. np.float64 and np.float32 correspond to double
    # and float types, respectively (in Java, C++, etc.); hence the error message, and reason for casting to float32.
    # N.b. this isn't an issue during development/training.
    processed_img = np.float32(img)

    # see comparison of different interpolation methods: <https://chadrick-kwag.net/cv2-resize-interpolation-methods/>
    processed_img = cv2.resize(processed_img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)

    # We can use TF's ResNet `preprocess_input`:
    # <https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/preprocess_input>; however, because
    # the datasets are quite different, it might do more harm than good <https://stackoverflow.com/a/58250681>; we can
    # therefore, simply divide by 255:
    processed_img = processed_img / 255

    # ResNet model expects input shape (batch_size, img_height, img_width, channels); therefore, we need to add
    # batch_size dimension:
    processed_img = np.expand_dims(processed_img, axis=0)
    return processed_img


def get_embedding(file_path: str):
    """ Get the embedding vector of a given image.

    :param file_path: File location of the image.
    :return {np.ndarray}: The embedding vector (extracted features) of the image after it goes through the network.
    """

    img = cv2.imread(file_path)
    img = process_image(img)
    embedding = model.predict(img)
    embedding = np.squeeze(embedding)  # ensure output is 1D array

    return embedding


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Calculate the embedding vectors and k-approximate-nearest-neighbor index:
# We will set up locality sensitive hashing to make our predictions more efficient. According to the Annoy documentation, there's a trade-off between the number of hyperplanes (`n_trees` arg) and the file size of the Annoy index. However, upon testing different values myself, I didn't notice much difference -- the size of the index file was pretty similar for this use-case.
#
# ##### Deciding on the number of hyperplanes when working on locality sensitive hashing:
# The Amazon Berkeley Objects (ABO) Dataset has more than 398,000 images.
# If we want each bucket to have 50 items, then $\frac{398,000}{50}=7,960$ buckets.
# Each hyperplane divides the space into 2. So, $2^{n}=7,960 \therefore n = \log_{2}7,960 \approx 13$. Thus, we need 13 hyperplanes per instance.

# + pycharm={"name": "#%%\n"}
NUM_HYPERPLANES = 13
# The last layer in model is the embedding vector, so we can grab the shape of that:
VECTOR_DIM = model.get_layer('global_max_pooling2d').output.shape[1] # each embedding is length 2048

# + pycharm={"name": "#%%\n"}
# %%time

# 398,000 items in a dict will take up about 19MB of RAM:
file_lookup = {}
# See Annoy docs: <https://github.com/spotify/annoy>
annoy_ = AnnoyIndex(VECTOR_DIM, 'angular')

# TODO: can we optimize/speed-up this, perhaps via batch processing?
for i, file_path_ in enumerate(file_paths):
    embedding_ = get_embedding(file_path_)

    annoy_.add_item(i, embedding_)
    file_lookup[i] = file_path_

# Build and save Annoy index:
annoy_.build(NUM_HYPERPLANES)
annoy_.save('../data/annoy_index/img_embedding.ann')


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Find Similar Images:

# + pycharm={"name": "#%%\n"}
def find_similar_images(img_path: str, num_results: int = 5):
    """ Get N similar images.

    :param img_path: File path of source image. Function finds similar product images based on this source image.
    :param num_results: The number of similar items we want to return. E.g. if set to 5, the function will return
        the closest 5 images.
    :return {list}: A list of file paths (strings).
    """

    # N.b. if user searches for an image that already exists in our dataset,
    # we DO want to return the exact same image; it's not a duplicate result.

    # Load single image, process, and get embedding:
    target_embedding = get_embedding(img_path)

    ids_of_matches = annoy_.get_nns_by_vector(target_embedding, num_results)

    top_matches = [file_lookup[id] for id in ids_of_matches]
    return top_matches


# + pycharm={"name": "#%%\n"}
def display_similar_images(img_path: str, num_results: int = 5):
    """ Get and display N similar images.

    :param img_path: File path of source image. Function finds similar product images based on this source image.
    :param num_results: The number of similar items we want to return. E.g. if set to 5, the function will return
        the closest 5 images.
    :return {None}: Renders images in the notebook.
    """

    list_images = find_similar_images(img_path, num_results)

    # display multiple images; see <https://stackoverflow.com/q/19471814>:
    for file_path in list_images: # recall `top_results` returns a tuple of (similarity, file_path)
        img = cv2.imread(file_path)
        plt.figure()
        file_name = file_path.split('/')[-1]
        plt.title(file_name)
        plt.imshow(img)


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Testing our model:
# Let’s test our model with a few example images to make sure we are getting the expected results.

# + pycharm={"name": "#%%\n"}
# N.b. change this file path to location of your example image:
example_img_path = '../data/2610.jpg'

print('----- Selected Image: -----')
plt.imshow(cv2.imread(example_img_path))

# + pycharm={"name": "#%%\n"}
print('----- Similar Images: -----')
display_similar_images(example_img_path)

# + pycharm={"name": "#%%\n"}

