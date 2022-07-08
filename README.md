# Product/Image Similarity
- Built with TensorFlow.
- Uses pretrained [ResNet model](https://arxiv.org/pdf/1512.03385.pdf) to create image embeddings. 
- Uses [Annoy package](https://github.com/spotify/annoy) for locality sensitive hashing (k-approximate-nearest-neighbors) to find similar product images quickly.

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

### Configurations:
- TensorFlow Model Serving: See official documentation on [TF serving config file](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/serving_config.md)
    - [Example implementation of models.config using docker-compose](https://stackoverflow.com/a/56590829) 

### License and Credits:
Copyright © 2022 Deepankara Reddy. BSD-2 license.

- [Amazon Berkley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) data from Amazon.com -- [CC BY-NC 4.0](https://amazon-berkeley-objects.s3.amazonaws.com/LICENSE-CC-BY-NC-4.0.txt) License
- Footer icons from [simple-icons](https://github.com/simple-icons/simple-icons) -- CC0 1.0 License
- "Box" icon from [Lukasz Adam](https://lukaszadam.com/illustrations) -- CC0 license (MIT license)
