# This script needs to run before any other operation and the first activity of the interface.
# It is recommended to run it manually, otherwise it will be run automatically when interface is launched for the first time.

import os
import json

from sbert import load_model, load_embeddings, image_clustering, image_cluster_labeling


def train_pipeline(model, config: dict):

    # ========== COMPUTE EMBEDDINGS ==========

    print('Loading embeddings...')
    corpus_names, corpus_embeddings = load_embeddings(
        data=config['images_folder']['path'],
        model=model,
        embeddings_path=config['embeddings']['path'],
        use_precomputed=config['embeddings']['use_precomputed_embeddings'],
        batch_size=config['embeddings']['batch_size'],
        data_type='image'
    )


    # ========== CLUSTERING ==========

    print('Clustering started...')
    clusters = image_clustering(
        corpus_names, corpus_embeddings,
        folder_file_path=config['images_folder']['path'],
        threshold=config['clustering']['threshold'],
        min_community_size=config['clustering']['min_community_size'],
        clusters_to_show=0,
        results_to_show=0
    )

    print('Cluster labeling started...')
    labels_all = json.load(open(config['classification_labels']['path'], encoding='utf-8'))
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
    # load config file
    PROJECT_CONFIG = 'config_unsplash.json'
    config = json.load(open(PROJECT_CONFIG))
    # load device-independent model
    model = load_model(model_folder=config['model']['folder'], model_name=config['model']['name'])
    
    for device in ['mobile', 'desktop']:
        # os-independent creation of paths
        for filetype in ['images_folder', 'embeddings', 'top_categories', 'classification_labels']:
            config[device][filetype]['path'] = os.path.join(config['data_folder'], config[device][filetype]['filename'])
        # execute pipeline
        print(f'> {device} pipeline execution...')
        print(train_pipeline(model, config[device]))
