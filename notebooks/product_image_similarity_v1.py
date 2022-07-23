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
#     display_name: Python 3
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
# 3. We'll create a DataFrame that contains the image's embedding as well as its location (file path).
# 4. With the embedding vectors of our images, we can use cosine similarity to ascertain how "closely related" two images are.
# 5. We will need a mechanism to compare the input image with the images in our dataset. The naive solution is to iterate through our embedding DataFrame and calculate the cosine similarity of each pair (i.e. the input image and one of the images from our dataset).
# 7. For a given input image, we would need to iterate through the entire list of cosine similarity scores and keep the `N` highest scores.
# 9. Finally, we will create a prediction function that returns a list of similar images for the given input. We'll then test our model on a few example product images to ensure it's working as expected.

# + pycharm={"name": "#%%\n"}
# Import libraries:

# Standard lib:
import heapq
import pathlib
import glob
import os

# ML/data science libraries:
import numpy as np
from scipy import spatial
import pandas as pd

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
    return glob.iglob('../data/product-dataset/**/*.jpg', recursive=True)

file_paths = get_file_paths()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Exploring the Data:
#
# Let's look at the typical images in our dataset.
# N.b. the dataset includes different size images and images with different number of channels (i.e. both RGB and grayscale) images.

# + pycharm={"name": "#%%\n"}
example_img = cv2.imread(next(file_paths))
print(plt.imshow(example_img))

# Reset generator:
file_paths = get_file_paths()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Load ResNet Model:
# Since we only need to generate the embedding vectors, we can make use of the well-known [ResNet model](https://arxiv.org/abs/1512.03385). ResNet is [already included in Keras/TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/ResNet50V2), and we can load pre-trained weights from ImageNet. In other words, we don't need to create a new model from scratch, and we don't need to do any training. We will remove the final couple layers in the network (i.e. the fully connected layer and the layer before that) -- these will later be replaced with a global max pooling layer, which is used for feature extraction.

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
def cosine_similarity(embedding_1: np.ndarray, embedding_2: np.ndarray):
    """ Calculates the cosine similarity of two vectors.

    :param embedding_1: An embedding vector.
    :param embedding_2: An embedding vector.
    :return: A single number (numpy float) -- the cosine similarity value.
    """

    # `distance.cosine` computes distance, not similarity; subtract by 1 to get similarity:
    return 1 - spatial.distance.cosine(embedding_1, embedding_2)

def process_image(img: np.ndarray):
    """ Pre-process images before feeding to model.

    Resizes image, scales (/255), and expands array dimension. The model requires specific input dimensions (shape),
    therefore resizing and adding dimension is necessary. Scaling improves performance.
    TODO: ideally, we should use standard deviation and mean of dataset instead of simply dividing by 255.

    :param img:
    :return {np.ndarray}: returns processed image
    """
    # see comparison of different interpolation methods: <https://chadrick-kwag.net/cv2-resize-interpolation-methods/>
    processed_img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)

    # We can use TF's ResNet `preprocess_input` <https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/preprocess_input>
    # however, because the datasets are quite different, it might do more harm than good <https://stackoverflow.com/a/58250681>;
    # we can therefore, simply divide by 255:
    processed_img = processed_img / 255

    # ResNet model expects input shape (batch_size, img_height, img_width, channels) -- we need to add batch_size dimension:
    processed_img = np.expand_dims(processed_img, axis=0)
    return processed_img


def get_embedding(file_path: str):
    """ Get the embedding vector of a given image.

    :param file_path: File location of the image.
    :return {np.ndarray}: The embedding vector (extracted features) of the image after it goes through the network.
    """

    img = cv2.imread(file_path)
    img = process_image(img)

    embedding = model.predict(img, verbose=False)
    return embedding


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Calculate the Embedding Vector and Cosine Similarity:
# We'll iterate thorough the entire dataset of images and append the image's embedding and file path as a new row in our DataFrame.

# + pycharm={"name": "#%%\n"}
df_embeddings = pd.DataFrame(columns=['file', 'file_path', 'embedding'])

# TODO: can we optimize/speed-up this, perhaps via batch processing?
for file_path_ in file_paths:
    embedding_ = get_embedding(file_path_)
    file_name_ = file_path_.split('/')[-1]

    # N.b. even though `embedding_` is already a 1xN vector, we need to wrap it in a list when building the DataFrame (per-column arrays must each be 1-D):
    df_embeddings = pd.concat([df_embeddings, pd.DataFrame({'file': file_name_, 'file_path': file_path_, 'embedding': [embedding_]})],
                              ignore_index=True)

# + pycharm={"name": "#%%\n"}
df_embeddings.head()

# + pycharm={"name": "#%%\n"}
# Write out to file:
df_embeddings.to_csv(pathlib.Path('../data/embedding.csv'))

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Tabularizing the cosine similarity scores:
#
# This is completely optional, but I wanted to better visualize the cosine similarity values across the entire dataset. In order to do so, I created an $N\times N$ matrix, where the image in each row is paired with every image (in each column).
#
# When we print out the matrix, notice the diagonal line of ones when we compare the similarity of an image to itself. This tells us that an image is perfectly similar to itself, which is expected.
#
# Another observation is that most similarity values are fairly high -- greater than $0.5$. This is due to the fact this particular dataset is made up of clothing items (shirts, shoes, dresses, etc.). Plus, the image quality and setting (professionally shot in front of a white background -- as opposed to user submissions from varying devices) all add to the consistency and similarity of the images.

# + pycharm={"name": "#%%\n"}
n_images = len(df_embeddings)
similarity_scores = np.zeros((n_images, n_images))
for i in range(n_images):
    for j in range(n_images):
        similarity_scores[i, j] = cosine_similarity(df_embeddings.iloc[i]['embedding'], df_embeddings.iloc[j]['embedding'])

# + pycharm={"name": "#%%\n"}
# Create empty DataFrame with file names as both the column and index names:
file_names = df_embeddings.loc[:, 'file'].tolist()
df_similarity = pd.DataFrame(similarity_scores, columns=file_names, index=file_names)

# Write out to file:
df_similarity.to_csv(pathlib.Path('../data/similarity_scores.csv'))

df_similarity.head()


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Find Similar Images:

# + pycharm={"name": "#%%\n"}
def find_most_similar_images(img_path: str, num_results: int = 5):
    """ Get the N most similar images.

    Since the function iterates through the entire dataset, we get the absolute best matches.

    :param img_path: File path of source image. Function finds similar product images based on this source image.
    :param num_results: The number of similar items we want to return. E.g. if set to 5, the function will return
        the closest 5 images.
    :return {list}: The top matches. Each element in list is a tuple of (similarity, file_path) where similarity
        is a float, and file_path is a string.
    """

    # Load single image, process, and get embedding:
    target_embedding = get_embedding(img_path)

    # Find max embedding:
    top_matches = []
    for db_img_file_path, db_img_embedding in zip(df_embeddings['file_path'], df_embeddings['embedding']):
        # top_matches needs to exclude the target image itself from being returned:
        if os.path.samefile(db_img_file_path, img_path):
            continue

        similarity_score = cosine_similarity(target_embedding, db_img_embedding)

        # We use `heapq` and keep only N number of elements -- this prevents us from holding the entire dataset in memory:
        # Ensure heap has N number of elements (this is done by adding the first N items):
        if len(top_matches) < num_results:
            heapq.heappush(top_matches, (similarity_score, db_img_file_path))
        # After creating an N-element heap, if a new item has a LARGER value than the SMALLEST in the heap,
        # then replace the smallest with the new item:
        # `heapq.nsmallest` returns a list, each element in list is a tuple (similarity, file_path); hence the reason for the double subscript `[0][0]`:
        elif similarity_score > heapq.nsmallest(1, top_matches)[0][0]:
            heapq.heapreplace(top_matches, (similarity_score, db_img_file_path))

    return top_matches


# + pycharm={"name": "#%%\n"}
def display_similar_images(img_path: str, num_results: int = 5):
    """ Get and display N similar images.

    :param img_path: File path of source image. Function finds similar product images based on this source image.
    :param num_results: The number of similar items we want to return. E.g. if set to 5, the function will return
        the closest 5 images.
    :return {None}: Renders images in the notebook.
    """

    top_results = find_most_similar_images(img_path, num_results)

    # Display multiple images; see <https://stackoverflow.com/q/19471814>:
    for similarity_score, file_path in top_results: # Recall `top_results` returns a tuple of (similarity, file_path)
        img = cv2.imread(file_path)
        plt.figure()
        plt.title(file_path.split('/')[-1]) # Use file name as figure title
        plt.imshow(img)


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Testing our model:
#
# Let's test our model with a few example images to make sure we are getting the expected results.

# + pycharm={"name": "#%%\n"}
example_img_path = '../data/product-dataset/2610.jpg'

print('----- Selected Image: -----')
plt.imshow(cv2.imread(example_img_path))

# + pycharm={"name": "#%%\n"}
print('----- Similar Images: -----')
display_similar_images(example_img_path)

# + pycharm={"name": "#%%\n"}

