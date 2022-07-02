# Product/Image Similarity
- Built with TensorFlow.
- Uses pretrained [ResNet model](https://arxiv.org/pdf/1512.03385.pdf) to create image embeddings. 
- Uses [Annoy package](https://github.com/spotify/annoy) for locality sensitive hashing (k-approximate-nearest-neighbors) to find similar product images quickly.

### TODO:
- [ ] Combine image similarity with text (product description) similarity.
- [ ] Replace cv2 with PIL (due to cv2 using BGR and PIL being more common in Tensorflow ecosystem)

### Limitations:
- The app only looks for image similarity, but for a more robust solution, we might want to take into account both image and the product description (if it exists).
- Annoy doesn't support incremental additions -- we can't add items once the index has been built. FAISS supports updatable indices, and would be a better choice for that case.
- Annoy file size grows quadratically with # of items in index.

### License and Credits:
Copyright © 2022 Deepankara Reddy. BSD-2 license.

- Footer icons from [simple-icons](https://github.com/simple-icons/simple-icons) - CC0 1.0 License
- Package/box icon from [Lukasz Adam](https://lukaszadam.com/illustrations) - CC0 license (MIT license)
