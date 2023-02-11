import faiss

def get_IndexFlatL2(size):
    return faiss.IndexFlatL2(size)  


def get_IndexLSH(size):
    return faiss.IndexLSH(size, 1500)


def get_IndexIVF(size, features):
    n_centroid = 50  # number of cells (centroids)
    
    quantizer = faiss.IndexFlatL2(size)
    index = faiss.IndexIVFFlat(quantizer, size, n_centroid)
    index.train(features)
    return index


def get_IndexPQ(size, features):
    n_subvectors = 128 # number of subvectors that each vector will be splitted into 
    bits = 8

    index = faiss.IndexPQ(size, n_subvectors, bits)
    index.train(features)

    return index