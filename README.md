# Product/Image Similarity
- Built with TensorFlow.
- Uses pretrained [ResNet model](https://arxiv.org/pdf/1512.03385.pdf) to create image embeddings. 
- Uses [Annoy package](https://github.com/spotify/annoy) for locality sensitive hashing (k-approximate-nearest-neighbors) to find similar product images quickly.

### TODO:
- [ ] Combine image similarity with text (product description) similarity.
- [ ] Replace cv2 with PIL (due to cv2 using BGR and PIL being more common in Tensorflow ecosystem)

### License:
Copyright © 2022 Deepankara Reddy. BSD-2 license.
