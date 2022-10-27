# Entity Embeddings for Entity Ranking: A Replicability Study

This repository is for the paper **Entity Embeddings for Entity Ranking: A Replicability Study** submitted to ECIR 2023 Reproducibility Track. 

## Entity Re-ranking Framework

This framework replicates the work of *Gerritse et al. (2020) Graph-Embedding Empowered Entity Retrieval*. The re-implementation code is found at Code/entity_reranking/entity_rerank.py that determines the embedding score between the entities linked in query and candidate set of entities. The steps to reproduce the replication study are given below.

***Please note that when you run the script below, the embeddings would be downloaded for Wikipedia2Vec, ERNIE and E-BERT which would take ~60 GB of space***

``` 
bash downloads.sh 
```

The script downloads.sh downloads the embeddings and rank-lips library and stores in the ```Data/embeddings``` folder and the rank-lips library is stored at ```Data/ranklips``` folder.

Next we generate the entity rankings with the embedding score between the candidate set of entities and entities linked in queries using Wikipedia2Vec, ERNIE and E-BERT entity embeddings.

``` 
bash entity_reranking.sh 
```

The above script generates the output files ```Output/entity_reranking/$dataset/$embeddings_rerank.run``` in the ```Output/entity_reranking``` folder



