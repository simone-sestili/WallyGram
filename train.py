# This script needs to run before any other operation and the first activity of the interface.
# It is recommended to run it manually, otherwise it will be run automatically when interface is launched for the first time.

import os
import json

from sbert import load_data, load_model, load_embeddings, image_clustering, image_cluster_labeling


PROJECT_CONFIG = 'config_unsplash.json'


def train_pipeline():

    # ========== INITIALIZATION ==========

    config = json.load(open(PROJECT_CONFIG))
    config['embeddings']['path'] = os.path.join(config['data']['folder'], config['embeddings']['filename'])
    config['top_categories']['path'] = os.path.join(config['data']['folder'], config['top_categories']['filename'])


    # ========== LOAD DATA ==========

    print('Loading data...')
    unzip_data_path = load_data(
        data_folder=config['data']['folder'],
        filename=config['data']['filename'],
        download_url=config['data']['download_url']
    )

    print('Loading model...')
    model = load_model(model_folder=config['model']['folder'], model_name=config['model']['name'])

    print('Loading embeddings...')
    corpus_names, corpus_embeddings = load_embeddings(
        data=unzip_data_path,
        model=model,
        embeddings_path=config['embeddings']['path'],
        use_precomputed=config['embeddings']['use_precomputed_embeddings'],
        download_url=config['embeddings']['download_url'],
        batch_size=config['embeddings']['batch_size'],
    )


    # ========== CLUSTERING ==========

    print('Clustering started...')
    clusters = image_clustering(
        corpus_names, corpus_embeddings,
        folder_file_path=unzip_data_path,
        threshold=config['clustering']['threshold'],
        min_community_size=config['clustering']['min_community_size'],
        clusters_to_show=0,
        results_to_show=0
    )

    print('Cluster labeling started...')
    labels_all = json.load(open(os.path.join(config['data']['folder'], config['classification_labels']['filename']), encoding='utf-8'))
    top_categories = image_cluster_labeling(
        clusters=clusters[:config['top_categories']['number']],
        labels=labels_all,
        corpus_embeddings=corpus_embeddings,
        model=model
    )
    
    with open(config['top_categories']['path'], 'w', encoding='utf-8') as f:
        json.dump(top_categories, f)
        
    return 'Training complete'


if __name__ == '__main__':
    print(train_pipeline())
