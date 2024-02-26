
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

<br>
<br>

# Evaluation

<table>
    <thead>
        <tr>
            <th>Feature Extraction</th>
            <th>SIAM</th>
            <th>Index</th>
            <th>MAP10</th>
            <th>MAP@50</th>
            <th>MAP@100</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=8>
              VGG16
            </td>
            <td>_</td>
            <td>FlatL2</td>
            <td>0.57</td>
            <td>0.47</td>
            <td>0.41</td>
        </tr>
        <tr>
            <td>_</td>
            <td>LSH</td>
            <td>0.82</td>
            <td>0.68</td>
            <td>0.62</td>
        </tr>
        <tr>
            <td>_</td>
            <td>IVF</td>
            <td>0.67</td>
            <td>0.57</td>
            <td>0.53</td>
        </tr>
        <tr>
            <td>_</td>
            <td>PQ</td>
            <td>0.71</td>
            <td>0.59</td>
            <td>0.53</td>
        </tr>
        <tr>
            <td>Triplet</td>
            <td>FlatL2</td>
            <td>0.81</td>
            <td>0.78</td>
            <td>0.77</td>
        </tr>
        <tr>
            <td>Triplet</td>
            <td>PQ</td>
            <td>0.81</td>
            <td>0.78</td>
            <td>0.77</td>
        </tr>
        <tr>
            <td>GeM + Contrastive</td>
            <td>FlatL2</td>
            <td>0.90</td>
            <td>0.79</td>
            <td>0.73</td>
        </tr>
         <tr>
             <td>GeM + Contrastive</td>
            <td>LSH</td>
            <td>0.88</td>
            <td>0.76</td>
            <td>0.70</td>
         </tr>
        <tr>
            <td rowspan=5>
              ResNet50
            </td>
            <td>triplet</td>
            <td>FlatL2</td>
            <td>0.93</td>
            <td>0.84</td>
            <td>0.78</td>
        </tr>
        <tr>
            <td>Triplet</td>
            <td>PQ</td>
            <td>0.86</td>
            <td>0.83</td>
            <td>0.82</td>
        </tr>
        <tr>
            <td>GeM + Contrastive</td>
            <td>FlatL2</td>
            <td>0.93</td>
            <td>0.84</td>
            <td>0.78</td>
        </tr>
        <tr>
            <td>>GeM + Contrastive</td>
            <td>PQ</td>
            <td>0.92</td>
            <td>0.83</td>
            <td>0.78</td>
        </tr>
        <tr>
            <td>>GeM + Contrastive</td>
            <td>LSH</td>
            <td>0.91</td>
            <td>0.81</td>
            <td>0.75</td>
        </tr>
       <tr>
            <td rowspan=2>
              DELF
            </td>
            <td>-</td>
            <td>FlatL2</td>
            <td>0.83</td>
            <td>0.73</td>
            <td>0.69</td>
        </tr>
        <tr>
            <td>-</td>
            <td>IVF</td>
            <td>0.84</td>
            <td>0.78</td>
            <td>0.76</td>
        </tr>
        <tr>
            <td>HOG</td>
            <td>-</td>
            <td>FlatL2</td>
            <td>0.36</td>
            <td>0.28</td>
            <td>0.24</td>
        </tr>
        <tr>
            <td>BoVW</td>
            <td>-</td>
            <td>FlatL2</td>
            <td>0.43</td>
            <td>0.32</td>
            <td>0.27</td>
        </tr>
    </tbody>
</table>



    
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

