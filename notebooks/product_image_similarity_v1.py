# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + pycharm={"name": "#%%\n"}
# standard lib:
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

# plotting and imaging:
import cv2
import matplotlib.pyplot as plt


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Load and Process Images:

# + pycharm={"name": "#%%\n"}
# n.b. we don't need to sort since we're not training the model:
def get_file_paths():
    """ Create generator of all image file paths.

    :return {generator}
    """

    # We could also have used `pathlib.Path().glob()`, but that returns a POSIX path rather than str:
    return glob.iglob('../data/e-commerce-product-images/**/*.jpg', recursive=True)

file_paths = get_file_paths()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Explore Data -- i.e. images:
# N.b. the dataset includes different size images and images with different number of channels (i.e. both RGB and grayscale) images.

# + pycharm={"name": "#%%\n"}
example_img = cv2.imread(next(file_paths))
print(plt.imshow(example_img))

# reset generator:
file_paths = get_file_paths()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Load ResNet Model:

# + pycharm={"name": "#%%\n"}
# constants:
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

# + pycharm={"name": "#%%\n"}
# Helper methods:

def cosine_similarity(embedding_1:np.ndarray, embedding_2:np.ndarray):
    """ Calculates the cosine similarity of two vectors.

    :param embedding_1: An embedding vector.
    :param embedding_2: An embedding vector.
    :return: A single number (numpy float) -- the cosine similarity value.
    """

    # `distance.cosine` computes distance, not similarity; subtract by 1 to get similarity:
    return 1 - spatial.distance.cosine(embedding_1, embedding_2)

def process_image(img:np.ndarray):
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

def get_embedding(file_path:str):
    img = cv2.imread(file_path)
    img = process_image(img)
    embedding = model.predict(img)

    return embedding


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Calculate the Embedding Vector and Cosine Similarity:

# + pycharm={"name": "#%%\n"}
# %%time

df_embeddings = pd.DataFrame(columns=['file', 'file_path', 'embedding'])

# TODO: can we optimize/speed-up this, perhaps via batch processing?
for file_path_ in file_paths:
    embedding_ = get_embedding(file_path_)
    file_name_ = file_path_.split('/')[-1]

    df_embeddings = df_embeddings.append({'file': file_name_, 'file_path': file_path_, 'embedding': embedding_}, ignore_index=True)

# + pycharm={"name": "#%%\n"}
# write out to file:
df_embeddings.to_csv(pathlib.Path('../data/output/embedding.csv'))

df_embeddings.head()

# + pycharm={"name": "#%%\n"}
# %%time

n_images = len(df_embeddings)
similarity_scores = np.zeros((n_images, n_images))
for i in range(n_images):
    for j in range(n_images):
        similarity_scores[i, j] = cosine_similarity(df_embeddings.iloc[i]['embedding'], df_embeddings.iloc[j]['embedding'])

# + pycharm={"name": "#%%\n"}
# create empty dataframe with file names as both the column and index names:
file_names = df_embeddings.loc[:, 'file'].tolist()
df_similarity = pd.DataFrame(similarity_scores, columns=file_names, index=file_names)

# write out to file:
df_similarity.to_csv(pathlib.Path('../data/output/similarity_scores.csv'))

df_similarity.head()


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Find Similar Images:

# + pycharm={"name": "#%%\n"}
def find_most_similar_images(img_path:str, num_results:int=5):
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
def display_similar_images(img_path:str, num_results:int=5):
    top_results = find_most_similar_images(img_path, num_results)

    # display multiple images; see <https://stackoverflow.com/q/19471814>:
    for similarity_score, file_path in top_results: # recall `top_results` returns a tuple of (similarity, file_path)
        img = cv2.imread(file_path)
        plt.figure()
        plt.title(file_path.split('/')[-1]) # use file name as figure title
        plt.imshow(img)


# + pycharm={"name": "#%%\n"}
example_img_path = '../data/e-commerce-product-images/Footwear/Men/Images/images_with_product_ids/3797.jpg'

print('----- Selected Image: -----')
plt.imshow(cv2.imread(example_img_path))

# + pycharm={"name": "#%%\n"}
print('----- Similar Images: -----')
display_similar_images(example_img_path)

# + pycharm={"name": "#%%\n"}

