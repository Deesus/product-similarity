# Related Product Recommender
###### An end-to-end machine learning application that recommends related product images.
___

Check out the [latest Jupyter Notebook](https://github.com/Deesus/product-similarity/blob/master/notebooks/product_image_similarity_v3.ipynb) for details/walkthrough on how the core elements of the model and app were developed.

#### Features Include:
- **Uploading a product image** returns a set of related product images.
- Users can **click on one of the results** to get a new set of recommended (similar) product images.
- Built on a dataset of **nearly 400,000 product images** from the [Amazon Berkley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) dataset.

#### Misc Info:
- Built with TensorFlow. 
- Uses pre-trained [ResNet](https://arxiv.org/pdf/1512.03385.pdf) model to create image embeddings. 
- Uses [Annoy package](https://github.com/spotify/annoy) for locality sensitive hashing (k-approximate-nearest-neighbors) to find similar product images quickly.

## Quickstart:
The fastest and easiest way to get the app running on your local machine is by pulling the production Docker image. This Docker image is built on the Amazon Berkley Objects dataset, and therefore the app will only serve images in that dataset. If you want to use your own dataset (images), make changes to the code, or make other configuration changes, you'll need to follow the [normal setup](https://github.com/Deesus/product-similarity#normal-setup) below.

1. Ensure that you have [Docker and Docker Compose installed](https://docs.docker.com/desktop/install/linux-install/) on your machine.
2. From the terminal, pull the Docker image: `$ docker pull deesus/product-similarity-backend:prod`
3. Also pull the TensorFlow Serving image: `$ docker pull tensorflow/serving:2.9.0`
4. `cd` into the `/web` folder and fire up the Docker services: `$ docker-compose -f docker-compose.prod.yml up`
5. Open your browser and go to `http://localhost`

## Normal Setup:
This project includes development and production versions of the web app. Both run on Docker. You will need to run your dataset (images) through the neural network and move/rename a few files.

1. Ensure that you have [Docker and Docker Compose installed](https://docs.docker.com/desktop/install/linux-install/) on your machine.
2. The database file (`.db`), Annoy Index file (`.ann`), and the saved model files are not version controlled, and therefore are absent from this repo. You will need to generate those files -- here's how:
3. Fire up Jupyter Notebooks and open the latest notebook (`product_image_similarity_v4.ipynb`) found in the `/notebooks` folder.
4. Follow the steps in the notebook.
5. In the _"Exploring the Data"_ section of the notebook, you will need to change the path name used in `get_file_paths()` to point to your images folder.
6. In the _"Load ResNet"_ section of the notebook, the model is saved to `models/resnet_similarity/[MODEL_VERSION]`. Copy the `resnet_similarity` folder to `/web/model_server/models` -- the full path should look like this: `/web/model_server/models/resnet_similarity/[VERSION_NUMBER]`
7. After you run your images through the neural network, the Annoy index and DB files are built. You'll then need to copy/move them to appropriate location. Copy to `img_embedding.ann` file to `/web/backend/model_client/annoy_index/`. And copy the `database.db` file to `/web/backend/db/`. _Aside: The reason these generated files aren't automatically written to the `/web` folder is to prevent unknowingly overriding the existing files (especially since they are not version controlled)._
8. From the terminal, `cd` into the `/web` folder. We'll now build the Docker services. There are two versions: one for production and one for development.
    - For the development version of the images, run `$ docker-compose -f docker-compose.dev.yml up`
    - For the production images, run `$ docker-compose -f docker-compose.prod.yml up`
9. Open your browser.
    - If you're running the development version, go to  `http://localhost:3000`
    - If you're running the production version, go to `http://localhost`
    - If deployed to production (e.g. cloud instance), simply enter your server's URL.

## Technologies Used:
- [TensorFlow](https://www.tensorflow.org/overview/)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [gRPC](https://grpc.io/)
- [Docker](https://docs.docker.com/)
- [Annoy](https://github.com/spotify/annoy#annoy)
- [NumPy](https://numpy.org/doc/stable/)
- [OpenCV](https://opencv.org/)
- [Vue.js (Nuxt)](https://nuxtjs.org/)
- [SQL (SQLite)](https://docs.python.org/3/library/sqlite3.html)
- [Gunicorn](https://gunicorn.org/)
- [Flask](https://flask.palletsprojects.com/en/2.1.x/)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable) (only in the notebooks)

## Notebooks:
- Notebooks located in `/notebooks` are versioned -- please view the latest version for the most refined, most performant methods/model.
- Notebooks have also been converted to Python files via [Jupytext](https://jupytext.readthedocs.io/en/latest/index.html). These `.py` files are only for comparing diffs.
- Due to the size of the Amazon Berkley Objects dataset, the notebooks were initially trained on the smaller [E-commerce Product Images](https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images) dataset for correctness before training on ABO data.

#### An overview of the notebook versions:
##### v1:
- Created initial model using pre-trained ResNet.
- Created method for calculating cosine similarity.
- Created methods for finding and displaying most similar images.
- Generated DataFrame to store embedding vectors for dataset.

##### v2:
- Replaced creating and iterating through the embedding DataFrame with Annoy index; thus greatly increasing performance.
- Replaced cosine similarity search with k-approximate nearest neighbors (Annoy).
- Refactored how `find_most_similar_images()` method obtains the best matches.

##### v3:
- Replaced dict lookup with an SQLite database for the product lookup.
- Added additional documentation.

## Limitations:
- The app only looks for image similarity, but for a more robust solution, we might want to take into account both image and the product description (if it exists).
- Annoy doesn't support incremental additions -- we can't add items once the index has been built. FAISS supports updatable indices, and would be a better choice for that case.
- Annoy file size grows quadratically with # of items in index.
- OpenCV should probably be replaced with PIL (due to OpenCV using BGR instead of RGB, and PIL being more common in the TensorFlow ecosystem).
- Images with transparency, e.g. PNG files with transparent backgrounds, will throw an error if fed into the model.


## License and Credits:
Copyright © 2022-2024 Deepankara Reddy. BSD-2 license.

- [Amazon Berkley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) data from Amazon.com -- [CC BY-NC 4.0](https://amazon-berkeley-objects.s3.amazonaws.com/LICENSE-CC-BY-NC-4.0.txt) License
- Footer icons from [simple-icons](https://github.com/simple-icons/simple-icons) -- CC0 1.0 License
- "Box" icon from [Lukasz Adam](https://lukaszadam.com/illustrations) -- CC0 license (MIT license)
- [E-commerce Product Images](https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images) (only used in the notebooks) -- CC0 license
