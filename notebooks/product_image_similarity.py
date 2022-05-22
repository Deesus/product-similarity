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

# ML/data science libraries:
import numpy as np
import tensorflow as tf
from scipy import spatial
import pandas as pd

# TenserFlow/Keras classes:
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.utils import image_dataset_from_directory, load_img
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Rescaling, Resizing, Reshape

# plotting and imaging:
import matplotlib.pyplot as plt
from PIL import Image

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Load and Process Images:

# + pycharm={"name": "#%%\n"}
data_dir = pathlib.Path('../data/e-commerce-product-images')

image_list = list(data_dir.glob('**/*.jpg'))
image_count = len(image_list)
print('image count:', image_count)

# + pycharm={"name": "#%%\n"}
# create dataset of images:

IMG_HEIGHT  = 224
IMG_WIDTH   = 224
BATCH_SIZE  = 1
VAL_SPLIT   = 0.2
SEED        = 7

train_set = image_dataset_from_directory(
    data_dir,
    validation_split=VAL_SPLIT,
    seed=SEED,
    subset='training',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_set = image_dataset_from_directory(
    data_dir,
    validation_split=VAL_SPLIT,
    seed=SEED,
    subset='validation',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# + pycharm={"name": "#%%\n"}
# explore data -- i.e. images:

# if we used PIL, we could do `PIL.Image.open(image_list[100])`:
example_img = np.array(Image.open(image_list[0]))
print(plt.imshow(example_img))

# + pycharm={"name": "#%%\n"}
# processed image:

# for single images:
resize_and_scale = Sequential([
    Resizing(IMG_HEIGHT, IMG_WIDTH),
    Rescaling(1.0/255)
])

# for image dataset from `image_dataset_from_directory` (which already resize images):
scale = Sequential([
    Rescaling(1.0/255)
])

plt.imshow(resize_and_scale(example_img))

# + pycharm={"name": "#%%\n"}
# currently the shape of images are (height, width, channels):
print('current shape of image:', example_img.shape)

# but we want to add an extra dimension for batch size, i.e. shape of (batch size, height, width, channels):
image_batch = np.expand_dims(example_img, axis=0)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Load ResNet Model:

# + pycharm={"name": "#%%\n"}
resnet_model = ResNet50V2(weights='imagenet')
resnet_model.summary()

# + pycharm={"name": "#%%\n"}
# Since we're getting the "prediction" layer (last layer of ResNet), the results of the both models will be exactly the same:
extracted_feature = Model(
    name='ResNet50V2_ExtractFeature',
    inputs=resnet_model.input,
    outputs=resnet_model.get_layer('predictions').output
)

# + pycharm={"name": "#%%\n"}
# %%time

# n.b. if you set shuffle to True for the dataset, then using `file_paths` (a fixed list) won't work:
file_paths = train_set.file_paths
df_embeddings = pd.DataFrame(columns=['file', 'file_path', 'embedding'])

for i, (batch, _) in enumerate(train_set):
    # # see <https://stackoverflow.com/a/63734183> on how to plot image from `image_dataset_from_directory`:
    # print(plt.imshow(np.squeeze(batch.numpy().astype("uint8"))))

    processed = scale(batch)
    # if we wanted to get the embedding/feature from a single image, we'd have to use `np.expand_dims`
    # since we are processing a batch, the input shape is the expected (batch-size, height, width, channels):
    embedding = extracted_feature.predict(processed)

    df_embeddings = df_embeddings.append({'file': file_paths[i].split('/')[-1], 'file_path': file_paths[i], 'embedding': embedding}, ignore_index=True)

df_embeddings.head()

# + pycharm={"name": "#%%\n"}
# %%time

n_images = len(df_embeddings)
similarity = np.zeros((n_images, n_images))
for i in range(n_images):
    for j in range(n_images):
        # `distance.cosine` computes distance, not similarity; subtract by 1 to get similarity:
        similarity[i, j] = 1 - spatial.distance.cosine(df_embeddings.iloc[i]['embedding'], df_embeddings.iloc[j]['embedding'])

# + pycharm={"name": "#%%\n"}
# create empty dataframe with file names as both the column and index names:
file_names = df_embeddings.loc[:, 'file'].tolist()
df_similarity = pd.DataFrame(similarity, columns=file_names, index=file_names)
df_similarity.head()


# + pycharm={"name": "#%%\n"}
def find_most_similar_imgs(img_path:str, num_results:int=5):
    # load single image, process, and get embedding:
    target_img = np.array(Image.open(img_path))
    target_img_processed = resize_and_scale(target_img)
    target_embedding = extracted_feature.predict(np.expand_dims(target_img_processed, axis=0))

    # find max embedding:    
    top_matches = []
    for db_img_file_path, db_img_embedding in zip(df_embeddings['file_path'], df_embeddings['embedding']):
        similarity = 1 - spatial.distance.cosine(target_embedding, db_img_embedding)

        # ensure heap has N number of elements (this is done by adding the first N items):
        if len(top_matches) < num_results:
            heapq.heappush(top_matches, (similarity, db_img_file_path))
        # after creating an N-element heap, if a new item has a LARGER value than the SMALLEST in the heap,
        # then replace the smallest with the new item:
        elif similarity > heapq.nsmallest(1, top_matches)[0][0]: # `heapq.nsmallest` returns a list, each element in list is a tuple (similarity, file_path); hence the reason for the double subscript `[0][0]`
            heapq.heapreplace(top_matches, (similarity, db_img_file_path))

    return top_matches

top_results = find_most_similar_imgs('../data/e-commerce-product-images/Footwear/Men/Images/images_with_product_ids/1637.jpg')


# + pycharm={"name": "#%%\n"}
def display_similar_imgs(img_path:str, num_results:int=5):
    top_results = find_most_similar_imgs(img_path)

    # display multiple images; see <https://stackoverflow.com/q/19471814>:
    for similarity_score, file_path in top_results: # recall `top_results` returns a tuple of (similarity, file_path)
        img = Image.open(file_path)
        plt.figure()
        plt.title(file_path.split('/')[-1]) # use file name as figure title
        plt.imshow(img)


# + pycharm={"name": "#%%\n"}
# example_img = '../data/e-commerce-product-images/Footwear/Men/Images/images_with_product_ids/1637.jpg'
example_img = '../data/e-commerce-product-images/Apparel/Boys/Images/images_with_product_ids/4201.jpg'

print('==================== original image: ====================')
plt.figure()
plt.title('4201.jpg')
plt.imshow(Image.open(example_img))

# + pycharm={"name": "#%%\n"}
print('==================== similar images: ====================')
display_similar_imgs(example_img)


# + pycharm={"name": "#%%\n"}
def similar_imgs(img1_path:str, img2_path:str):
    img1 = np.array(Image.open(img1_path))
    img2 = np.array(Image.open(img2_path))

    img1 = resize_and_scale(img1)
    img2 = resize_and_scale(img2)

    img1_embedding = extracted_feature.predict(np.expand_dims(img1, axis=0))
    img1_embedding = extracted_feature.predict(np.expand_dims(img2, axis=0))

    similarity = 1 - spatial.distance.cosine(img1_embedding, img1_embedding)

    return similarity


def similarity_values(img1_path:str, img2_path:str):
    img1_name = img1_path.split('/')[-1]
    img2_name = img2_path.split('/')[-1]

    results = df_similarity.loc['50174.jpg'].to_numpy()
    return results[results == 1]


# + pycharm={"name": "#%%\n"}
similarity_values(
    '../data/e-commerce-product-images/Apparel/Boys/Images/images_with_product_ids/4201.jpg',
    '../data/e-commerce-product-images/Footwear/Women/Images/images_with_product_ids/41669.jpg'
)

# + pycharm={"name": "#%%\n"}
df_similarity.head()

# + pycharm={"name": "#%%\n"}
plt.figure()
plt.imshow(Image.open('../data/e-commerce-product-images/Apparel/Boys/Images/images_with_product_ids/4201.jpg'))
plt.figure()
plt.imshow(Image.open('../data/e-commerce-product-images/Footwear/Women/Images/images_with_product_ids/41669.jpg'))

print('similarity:',
      similar_imgs(
        '../data/e-commerce-product-images/Apparel/Boys/Images/images_with_product_ids/4201.jpg',
        '../data/e-commerce-product-images/Footwear/Women/Images/images_with_product_ids/41669.jpg'
      )
)

# + pycharm={"name": "#%%\n"}

