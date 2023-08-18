
# Landmark Image Retrieval

A project make use of deep learning and features extraction techniques to detect retrieve image

# Indexing
Faiss (Faiss is a library for efficient similarity search and clustering of dense vectors - developed primarily at Meta's Fundamental AI Research group) is use for indexing. Below are indexing technique which are used in project
* Flat Euclidean Distance
* Inverted File Index
* Locality Sensitive Hashing
* Product Quantization

# Feature extractions
* Bag of visual word (BOVW)

<p align="center">
<img src="https://github.com/ptatien0307/image_retrieval/assets/79583501/b3c96a7c-b333-49fb-bcbb-46ed742333af.png" alt="drawing" width="75%" height="75%"/>
</p>

* Histogram of oriented gradients (HOG)
  
* Use deep model to extract features
  
<p align="center">
<img src="https://github.com/ptatien0307/image_retrieval/assets/79583501/3622ffa9-b5b9-423a-8678-523a28dabd80.png" alt="drawing" width="50%" height="50%"/>
</p>

# Model

* VGG16
* ResNet50
* DELF (for more infomation: https://arxiv.org/abs/1612.06321)
<p align="center">
<img src="https://github.com/ptatien0307/image_retrieval/assets/79583501/2f56e48a-b740-4d86-b668-b7d35565680f0.png" alt="drawing" width="50%" height="50%"/>
</p>


* Also make use of some technique for retrieve such as:
  * GeM pooling (for more infomation: https://amaarora.github.io/posts/2020-08-30-gempool.html)

  <p align="center">
  <img src="https://github.com/ptatien0307/image_retrieval/assets/79583501/070b2c46-8936-4517-a74f-3fb6abb8cfc5.png" alt="drawing" width="50%" height="50%"/>
  </p>

  * Siamese network (for more information: https://en.wikipedia.org/wiki/Siamese_neural_network)
    * Triplet loss
      
    <p align="center">
    <img src="https://github.com/ptatien0307/image_retrieval/assets/79583501/68ec9f8b-7c7f-42fe-b4f6-4746c39ca70f.png" alt="drawing" width="50%" height="50%"/>
    </p>

    <p align="center">
    <img src="https://github.com/ptatien0307/image_retrieval/assets/79583501/76295dad-4ce7-48b2-8cb5-d5a68102b5f3.png" alt="drawing" width="50%" height="50%"/>
    </p>

    * Contrastive loss
    <p align="center">
    <img src="https://github.com/ptatien0307/image_retrieval/assets/79583501/b3ad4269-6694-41e5-8d21-3209c6ca5afb.png" alt="drawing" width="50%" height="50%"/>
    </p>
    <p align="center">
    <img src="https://github.com/ptatien0307/image_retrieval/assets/79583501/c8491dc4-6bf7-491a-93f4-d89744dc7bf7.png" alt="drawing" width="50%" height="50%"/>
    </p>
    
# Dataset
The dataset used for this project is a small part of the GLR21 (https://www.kaggle.com/competitions/landmark-recognition-2021). Here's some example
<p align="center">
<img src="https://github.com/ptatien0307/image_retrieval/assets/79583501/54aa62b0-de34-45a1-b11a-da70db0aa2ad.png" alt="drawing" width="50%" height="50%"/>
</p>

# Website demo
Using streamlit to build a website demo
User will choose which apart (whole image) to retrive
<p align="center">
<img src="https://github.com/ptatien0307/image_retrieval/assets/79583501/d9b61417-1fb6-4c15-aa3f-e40acd987453.png" alt="drawing" width="75%" height="75%"/>
</p>

Result will be return as top K (user will cho K result)
<p align="center">
<img src="https://github.com/ptatien0307/image_retrieval/assets/79583501/1287fa6e-d592-4a94-a82c-11d96a8a9a03.png" alt="drawing" width="75%" height="75%"/>
</p>

