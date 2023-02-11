import faiss
from indexer.get_indexer import get_IndexFlatL2
from indexer.get_indexer import get_IndexLSH
from indexer.get_indexer import get_IndexIVF
from indexer.get_indexer import get_IndexPQ

def indexing(features, name, index_type):
    # Build the index
    if index_type == 1:
        faiss_indexer = get_IndexFlatL2(features[0].shape[0])   
    elif index_type == 2:
        faiss_indexer = get_IndexLSH(features[0].shape[0])   
    elif index_type == 3:
        faiss_indexer = get_IndexIVF(features[0].shape[0], features)   
    elif index_type == 4:
        faiss_indexer = get_IndexPQ(features[0].shape[0], features)  
        
    faiss_indexer.add(features)

    faiss.write_index(faiss_indexer, f'/content/drive/MyDrive/cs336/indexes/{name}.index.bin')
