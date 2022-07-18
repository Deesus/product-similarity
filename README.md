# Product/Image Similarity
###### An end-to-end machine learning application that displays related images.

A deployed version is available at: [product-similarity.deepankara.com](https://product-similarity.deepankara.com)

##### Features Include:
- **Uploading a product image** returns a set of similar product images.
- Users can **click on one of the results** to get a new set of similar images.
- Built on a dataset of **over 398,000 product images** from the [Amazon Berkley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) dataset.

##### Misc Info:
- Check out the latest Jupyter Notebook in `/notebooks` for details/walkthrough on how the core elements of the model and app were developed.
- Built with TensorFlow. Uses pretrained [ResNet](https://arxiv.org/pdf/1512.03385.pdf) model to create image embeddings. 
- Uses [Annoy package](https://github.com/spotify/annoy) for locality sensitive hashing (k-approximate-nearest-neighbors) to find similar product images quickly.

### Quickstart:
This project includes development and production versions of the web app. Both run on Docker. You will need to run your dataset (images) through the neural network and move/rename a few files. See section, _"Running the Web App With Custom Data"_, below for more details.

1. Ensure you have [Docker and Docker Compose installed](https://docs.docker.com/desktop/install/linux-install/) on your machine.
2. If you're running the development Docker images or want to use your own dataset, follow the steps in _"Running the Web App With Custom Data"_ below before continuing.
3. From the terminal, `cd` into the `/web` folder. This is the location of the entire web app and its dependencies.
4. If you deployed the repo to a production environment (e.g. cloud instance), then you'll need to add your domain URL to the CORS list in `/web/routes/routes.py`.
5. We'll now build the Docker services. There are two versions: one for production and one for development.
   - For the production images, run `$ docker-compose -f docker-compose.prod.yml up`
   - For the development version of the images, run `$ docker-compose -f docker-compose.dev.yml up`
5. Open your browser and go to `http://localhost:3000`. (If deployed to production, simply enter your server's URL.)

##### Running the Web App With Custom Data:
You can use your own dataset (images) for the product-similarity web app. The database file (`.db`), Annoy Index file (`.ann`), and the saved model files are not version controlled, and therefore are absent from this repo. You will need to generate those files -- here's how:

1. Fire up Jupyter Notebooks and open the latest notebook (`product_image_similarity_v4.ipynb`) found in the `/notebooks` folder .
2. Follow the steps in the notebook:
3. In the _"Exploring the Data"_ section of the notebook, you will need to change the path name used in `get_file_paths()` to point to your image folder.
4. In the _"Load ResNet"_ section of the notebook, the model is saved to `models/resnet_similarity/[MODEL_VERSION]`. Copy the `resnet_similarity` folder to `/web/model_server/models` -- the full path should look like this: `/web/model_server/models/resnet_similarity/[VERSION_NUMBER]`.
5. After you run your images through the neural network, the Annoy index and DB files are built. You'll then need to copy/move them to appropriate location. Copy to `img_embedding.ann` file to `/web/backend/model_client/annoy_index/`. And copy the `database.db` file to `/web/backend/db/`.

N.b. the reason these generated files aren't automatically written to the `/web` folder is to prevent unknowingly overriding the existing files (especially since they are not version controlled).

### Technologies Used:
- [TensorFlow](https://www.tensorflow.org/overview/)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [gRPC](https://grpc.io/)
- [Docker](https://docs.docker.com/)
- [Annoy](https://github.com/spotify/annoy#annoy)
- [NumPy](https://numpy.org/doc/stable/)
- [Vue.js (Nuxt)](https://nuxtjs.org/)
- [SQL (SQLite)](https://docs.python.org/3/library/sqlite3.html)
- [Node.js](https://nodejs.org/)
- [Gunicorn](https://gunicorn.org/)
- [Flask](https://flask.palletsprojects.com/en/2.1.x/)

### TODO:
- [ ] Combine image similarity with text (product description) similarity.
- [ ] Replace cv2 with PIL (due to cv2 using BGR and PIL being more common in Tensorflow ecosystem)

### Limitations:
- The app only looks for image similarity, but for a more robust solution, we might want to take into account both image and the product description (if it exists).
- Annoy doesn't support incremental additions -- we can't add items once the index has been built. FAISS supports updatable indices, and would be a better choice for that case.
- Annoy file size grows quadratically with # of items in index.

### License and Credits:
Copyright © 2022 Deepankara Reddy. BSD-2 license.

- [Amazon Berkley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) data from Amazon.com -- [CC BY-NC 4.0](https://amazon-berkeley-objects.s3.amazonaws.com/LICENSE-CC-BY-NC-4.0.txt) License
- Footer icons from [simple-icons](https://github.com/simple-icons/simple-icons) -- CC0 1.0 License
- "Box" icon from [Lukasz Adam](https://lukaszadam.com/illustrations) -- CC0 license (MIT license)
