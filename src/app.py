import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
import os
from search import search, load_index, load_extractor
from imutils import paths
import numpy as np

st.set_page_config(page_title="Upload Image", page_icon=":camera:", layout="wide")

def load_image(image_file):
    img = Image.open(image_file)
    return img

def upload_image(img_file, box_color, search_btn):
    # print query image and cropped image
    if img_file:
        img = Image.open(img_file)
        st.header("Uploaded image")
        # Get a cropped image from the frontend
        cropped_img = st_cropper(img, realtime_update=True, box_color=box_color,
                                 aspect_ratio=None)

        # Manipulate cropped image at will
        st.header("Preview query image")
        st.image(cropped_img)
        if search_btn:
            cropped_img.save("cropped.jpg")
    return None

def plot(top_k, result):
    st.header("Results")
    col1, col2, col3, col4, col5 = st.columns(5)
    for i in range(top_k//5):
        img1 = load_image(result[5*i + 0])
        img2 = load_image(result[5*i + 1])
        img3 = load_image(result[5*i + 2])
        img4 = load_image(result[5*i + 3])
        img5 = load_image(result[5*i + 4])
        with col1:
            st.image(img1, use_column_width=True)
        with col2:
            st.image(img2, use_column_width=np.True_)
        with col3:
            st.image(img3, use_column_width=True)
        with col4:
            st.image(img4, use_column_width=True)
        with col5:
            st.image(img5, use_column_width=True)


    st.write(f"Top {top_k} results using {method} method")

def handle_search(search_btn, query, index, extractor, collection_paths, top_k):
    # Save cropped image
    if search_btn:
        result = search(query, index, extractor, collection_paths, top_k)
        plot(top_k, result)


def load_item(method):
    collection_paths = np.array(list(paths.list_images('/content/drive/MyDrive/cs336/dataset/collection')))
    index = None
    extractor = None
    
    
    if method == 'DELF':
        # Sort collection paths if using DELF
        collection_paths = sorted(collection_paths, key=lambda x: x.split('/')[-1])
        collection_paths = np.array(collection_paths)

        # Load index and extractor
        index = load_index('/content/drive/MyDrive/cs336/indexes/DELF_IVF.index.bin')
        extractor = load_extractor('/content/drive/MyDrive/cs336/indexes/DELF_IVF.index.bin')
    elif method == 'ResNet50_TripletLoss':
        # Load index and extractor
        index = load_index('/content/drive/MyDrive/cs336/indexes/ResNet50_SIAM_PQ.index.bin')
        extractor = load_extractor('/content/drive/MyDrive/cs336/indexes/ResNet50_SIAM_PQ.index.bin')
    else:
        # Load index and extractor
        index = load_index('/content/drive/MyDrive/cs336/indexes/ResNet50_SIAM_GeM_PQ.index.bin')
        extractor = load_extractor('/content/drive/MyDrive/cs336/indexes/ResNet50_SIAM_GeM_PQ.index.bin')

    return index, extractor, collection_paths

if __name__ == "__main__":
    img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
    box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
    top_k = st.sidebar.slider("Top K", 5, 100, 5, 5)
    method = st.sidebar.selectbox("Method", ["DELF", "ResNet50_TripletLoss", "ResNet50_GeM_ContrastiveLoss"])
    search_btn = st.sidebar.button("Search")

    upload_image(img_file, box_color, search_btn)
    index, extractor, collection_paths = load_item(method)
    

    query = '/content/drive/MyDrive/cs336/cropped.jpg'
    handle_search(search_btn, query, index, extractor, collection_paths, top_k)




