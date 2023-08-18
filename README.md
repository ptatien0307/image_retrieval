
# Landmark Image Retrieval

A project make use of deep learning and features extraction techniques to detect retrieve image

## Model
* VGG16
* ResNet50
* DELF
* Also make use of some technique for retrieve such as: siamese network, GeM pooling
## Feature extractions
* Bag of visual word (BOVW)
* Histogram of oriented gradients (HOG)
* Use deep model to extract features
## Indexing
Faiss (Faiss is a library for efficient similarity search and clustering of dense vectors - developed primarily at Meta's Fundamental AI Research group) is use for indexing. 
* Flat Euclidean Distance
* Inverted File Index
* Locality Sensitive Hashing
* Product Quantization
![image](https://github.com/ptatien0307/image_retrieval/assets/79583501/67d52b71-ce65-4a6d-a00a-f07ca4b80575)
