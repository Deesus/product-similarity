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

# + pycharm={"name": "#%%\n"}
# standard lib:
import glob
import sqlite3

# ML/data science libraries:
import numpy as np
from annoy import AnnoyIndex

# TenserFlow/Keras classes:
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalMaxPool2D

# plotting and imaging:
import cv2
import matplotlib.pyplot as plt


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Get Files:

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

# + pycharm={"name": "#%%\n"}
# Save model:

# N.b. we need to specify model version as the file directory, otherwise TensorFlow Serving will
# throw an error that there aren't any servable model versions: <https://stackoverflow.com/q/45544928>:
MODEL_VERSION = 4
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

# + pycharm={"name": "#%%\n"}
# Helper methods:

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
    img = cv2.imread(file_path)
    img = process_image(img)
    embedding = model.predict(img)
    embedding = np.squeeze(embedding)  # ensure output is 1D array

    return embedding


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Setup lookup table for product images

# + pycharm={"name": "#%%\n"}
def get_db_connection():
    conn = sqlite3.connect('../data/database.db')
    conn.row_factory = sqlite3.Row
    return conn


# + pycharm={"name": "#%%\n"}
# Create table:
conn_ = get_db_connection()
cur = conn_.cursor()
cur.execute("""
    CREATE TABLE products (
        id INTEGER PRIMARY KEY UNIQUE,
        file_path TEXT NOT NULL
    )
""")
conn_.commit()
conn_.close()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Build Step: calculate the embedding vectors and k-approximate-nearest-neighbor index
#
# The Amazon Berkeley Objects (ABO) Dataset has more than 147,000 products.
# If we want each bucket to have 20 items, then $\frac{147,000}{20}=7,350$ buckets.
# Each hyperplane divides the space into 2. So, $2^{n}=7,350 \therefore n = \log_{2}7,350 \approx 13$. Thus, we need 13 hyperplanes per instance.

# + pycharm={"name": "#%%\n"}
NUM_HYPERPLANES = 13
# The last layer in model is the embedding vector, so we can grab the shape of that:
VECTOR_DIM = model.get_layer('global_max_pooling2d').output.shape[1] # each embedding is length 2048

# + pycharm={"name": "#%%\n"}
# %%time

# See Annoy docs: <https://github.com/spotify/annoy>
annoy_ = AnnoyIndex(VECTOR_DIM, 'angular')

# Open connection before for loop to reduce the overhead of opening and closing connection after every iteration in loop:
conn_ = get_db_connection()
cur = conn_.cursor()

# TODO: can we optimize/speed-up this, perhaps via batch processing?
for i, file_path_ in enumerate(file_paths):
    embedding_ = get_embedding(file_path_)
    annoy_.add_item(i, embedding_)

    # Add file path to db:
    cur.execute("INSERT INTO products (id, file_path) VALUES (?, ?)", (i, file_path_))

# Committing after each insertion will be safer than committing everything at once,
# but if any of the transactions fail during the loop, we'll have to start over anyway;
# so, we commit outside the loop, which is faster:
conn_.commit()
conn_.close()

# + pycharm={"name": "#%%\n"}
# Build and save Annoy index:
annoy_.build(NUM_HYPERPLANES)
annoy_.save('../data/annoy_index/img_embedding.ann')


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Find Similar Images:

# + pycharm={"name": "#%%\n"}
def query_product_by_id(id: int, cursor):
    """ Fetch SQL row object from db.

    Remember that we created a "dictionary cursor" (therefore we need to use subscript the get the column).

    :param id:
    :param cursor:
    :return {sqlite3.Row}
    """
    return cursor.execute('SELECT * FROM products WHERE id=?', (id,)).fetchone()


# + pycharm={"name": "#%%\n"}
def find_similar_images(img_path: str, num_results: int = 5):
    # N.b. if user searches for an image that already exists in database,
    # we DO want to return the exact same image; it's not a duplicate result.

    # Load single image, process, and get embedding:
    target_embedding = get_embedding(img_path)

    ids_of_matches = annoy_.get_nns_by_vector(target_embedding, num_results)

    # Get file paths from lookup table:
    # TODO: we should create a decorator for opening/closing db connection:
    conn = get_db_connection()
    cursor = conn.cursor()
    top_matches = [query_product_by_id(id, cursor)['file_path'] for id in ids_of_matches]
    cursor.close()

    return top_matches


# + pycharm={"name": "#%%\n"}
def display_similar_images(img_path: str, num_results: int = 5):
    list_images = find_similar_images(img_path, num_results)

    # display multiple images; see <https://stackoverflow.com/q/19471814>:
    for file_path in list_images: # recall `top_results` returns a tuple of (similarity, file_path)
        img = cv2.imread(file_path)
        plt.figure()
        file_name = file_path.split('/')[-1]
        plt.title(file_name)
        plt.imshow(img)


# + pycharm={"name": "#%%\n"}
example_img_path = '../data/e-commerce-product-images/Footwear/Women/Images/images_with_product_ids/2610.jpg'

print('----- Selected Image: -----')
plt.imshow(cv2.imread(example_img_path))

# + pycharm={"name": "#%%\n"}
print('----- Similar Images: -----')
display_similar_images(example_img_path)

# + pycharm={"name": "#%%\n"}

