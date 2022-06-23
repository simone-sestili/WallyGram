import os
import glob
import re
import time
import torch
import zipfile

from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from utils import load_pickle, dump_pickle, argmax_dict


# import win32file
# win32file._setmaxstdio(32768)


# ========== PRE-PROCESSING ==========

def pipeline(text: str, strings_to_exclude: list = []) -> str:
    """
    Clean given line from special characters
    """
    # remove lines with only dashes
    if re.match(r"^\s*-+\s*$", text):
        return ''
    # remove lines with pip install
    if re.match(r"pip install .+?", text):
        return ''
    # remove html tags
    text = re.sub(r"<.+?>", '', text)
    # remove urls
    text = re.sub(r"http\S+", '', text)
    # remove chinese characters
    text = re.sub(r"[\u4e00-\u9fff]+", '', text)
    
    text = text.lower().strip()
    
    # remove useless strings
    for excl in strings_to_exclude:
        if text.startswith(excl):
            return ''

    return text



# ========== LOAD FUNCTIONS ==========

def load_data(data_folder: str, filename: str, download_url: str = ''):
    """
    Load data in filename, if it needs to be downloaded it creates the folder, download the file and load/unzip the result.
    If the result is not a structured object for text (image scenario) return the path of the folder containing the images.
    """
    data_file_path = os.path.join(data_folder, filename)

    # creates data folder
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)

    # download from web
    if download_url:
        # check if data is already present
        if not os.path.exists(data_file_path):
            util.http_get(download_url, data_file_path)
    
    # decide load method depending on extension
    ext = data_file_path.split('.')[-1]
    
    if ext == 'zip':
        unzip_data_path = '.'.join(data_file_path.split('.')[:-1])
        if not os.path.exists(unzip_data_path):
            # unzip data
            os.makedirs(unzip_data_path)
            with zipfile.ZipFile(data_file_path, 'r') as zf:
                for member in tqdm(zf.infolist(), desc='Extracting'):
                    zf.extract(member, unzip_data_path)
        print(f'Loaded {len(os.listdir(unzip_data_path))} items')
        return unzip_data_path
    else:
        print(f'File {filename} is not currently supported')
        return None
    
    
def load_model(model_folder: str, model_name: str):
    """
    If model is in local files load it from there, otherwise download from HuggingFace
    """
    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)

    if model_name.startswith('cross-encoder'):
        # cross-encoder
        if model_name not in os.listdir(model_folder):
            model = CrossEncoder(model_name)
            model.save(os.path.join(model_folder, model_name))
        else:
            model = CrossEncoder(os.path.join(model_folder, model_name))
        return model
    else:
        # bi-encoder
        if model_name not in os.listdir(model_folder):
            model = SentenceTransformer(model_name)
            model.save(os.path.join(model_folder, model_name))
        else:
            model = SentenceTransformer(os.path.join(model_folder, model_name))
        return model


def load_embeddings(data, model, embeddings_path: str, use_precomputed: bool = False, download_url: str = '', batch_size: int = 32, data_type: str = 'text'):
    """
    This function loads and returns the aligned couple corpus_names, corpus_embeddings.
    - If user decides to use the precomputed embeddings the functions tries to load them from local storage or 
      from web, if this process fails it goes to the manual creation.
    - If embeddings must be manually created the process depends on the data_type, which can be either text or image. If the data is text based
      it has to be passed as a list of strings, it the data is image based it has to be passed the path of folder containing all the images.
    """
    if use_precomputed:
        if os.path.exists(embeddings_path):
            # load local embeddings
            corpus_names, corpus_embeddings = load_pickle(embeddings_path)
            print('Items:', len(corpus_names))
            return corpus_names, corpus_embeddings
        elif download_url:
            # download from web and store into embeddings_path
            util.http_get(download_url, embeddings_path)
            try:
                corpus_names, corpus_embeddings = load_pickle(embeddings_path)
                print('Items:', len(corpus_names))
                return corpus_names, corpus_embeddings
            except:
                pass  # something went wrong during download, re-compute embeddings
    if data_type == 'text' and type(data) == list:  # assume data is an ordered list of strings
        corpus_names = data
        print('Paragraphs:', len(corpus_names))
        corpus_embeddings = model.encode(
            data,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )
    elif data_type == 'image' or type(data) == str:  # assume data is images path
        corpus_names = list(glob.glob(data + '/*'))
        print('Images:', len(corpus_names))
        
        # if there are more than 1000 images then shard encodings
        num_shards = int(len(corpus_names) / 1000) + 1
        
        corpus_embeddings = model.encode(
            [Image.open(filepath) for filepath in corpus_names[(num_shards-1)*1000:]],
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        for shard in tqdm(range(1, num_shards)):            
            tmp = model.encode(
                [Image.open(filepath) for filepath in corpus_names[(shard-1)*1000:shard*1000]],
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            corpus_embeddings = torch.cat((corpus_embeddings, tmp), dim=-2)

    print('corpus_names:', len(corpus_names))   
    print('corpus_embeddings:', len(corpus_embeddings), corpus_embeddings.shape)
    dump_pickle((corpus_names, corpus_embeddings), embeddings_path)
    return corpus_names, corpus_embeddings



# ========== TASKS IMPLEMENTATION ==========

def image_clustering(corpus_names: list, corpus_embeddings: list, folder_file_path: str, threshold: float = 0.75, min_community_size: int = 10, clusters_to_show: int = 10, results_to_show: int = 3):
    """
    Fast method to create clusters of images with similar meaning.
    The threshold value controls the selectivity to differentiate different clusters.
    Shows in a jupyter notebook some images from some clusters. 
    """
    clusters = util.community_detection(corpus_embeddings, threshold=threshold, min_community_size=min_community_size)
    for cluster in clusters[:clusters_to_show]:
        from IPython.display import display
        from IPython.display import Image as IPImage
        print("\n\nCluster size:", len(cluster))
        # output first 3 images
        for idx in cluster[:results_to_show]:
            display(IPImage(os.path.join(folder_file_path, corpus_names[idx]), width=200))
        time.sleep(0.1)
    
    return clusters


def image_classification(corpus_names: list, corpus_embeddings: list, folder_file_path: str, labels: list, model, results_to_show: int = 10):
    """
    Classified a corpus of images converted into word embeddings within a given set of labels.
    Shows a given number of results in a jupyter notebook.
    """
    # convert labels to word embeddings
    labels_embeddings = model.encode(labels, convert_to_tensor=True)
    
    # compute cross scores through cosine similarity
    cos_scores = util.cos_sim(corpus_embeddings, labels_embeddings)
    
    # extracts highest cosine similarity for each image
    pred_labels = torch.argmax(cos_scores, dim=1)
    
    # show results
    for idx in range(results_to_show):
        from IPython.display import display
        from IPython.display import Image as IPImage
        display(IPImage(os.path.join(folder_file_path, corpus_names[idx]), width=200))
        print(f'Predicted label: {labels[pred_labels[idx]]}\n\n')
        time.sleep(0.1)
    
    return [labels[pred] for pred in pred_labels]


def image_cluster_labeling(clusters: list, labels: list, corpus_embeddings: list, model) -> list:
    """
    Given some sets of already clusterized similar images (each cluster is the list of the corpus ids)
    the function searches for the best label (amongst the given list) for each cluster, given that the
    best label is the one that maximizes the average cosine similarity
    """
    res = []

    for cluster in tqdm(clusters):
        label_similarity = {}
        for label in labels[:100]:
            # convert label to word embedding
            label_embeddings = model.encode([label], convert_to_tensor=True)
            # compute cross scores through cosine similarity of label all images in given cluster
            cross_scores = util.cos_sim(corpus_embeddings[cluster], label_embeddings)
            # assign to given label the average cosine similarity on the whole cluster
            label_similarity[label] = sum(cross_scores) / len(cross_scores)
        # get label with the highest similarity
        res.append(argmax_dict(label_similarity))
        
    return list(set(res))


def text2image(query: str, image_names: list, image_embeddings: list, folder_file_path: str, img_model, top_k: int = 5, DEBUG: bool = False):
    """
    Given a textual query it return a set of images with the most similar meaning
    """
    # convert query to word embedding using CLIP
    query_embedding = img_model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    # search most similar images
    results = util.semantic_search(query_embedding, image_embeddings, top_k=top_k)
    results = results[0]  # get result of the first query
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    if DEBUG:
        from IPython.display import display
        from IPython.display import Image as IPImage
        # show results
        print('Query:')
        display(query)
        for res in results:
            print('Score', round(res['score'], 3))
            display(IPImage(os.path.join(folder_file_path, image_names[res['corpus_id']]), width=200))
            time.sleep(0.1)
    
    return results
