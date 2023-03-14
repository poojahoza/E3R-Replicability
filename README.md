# Entity Embeddings for Entity Ranking: A Replicability Study

This repository is for the paper **Entity Embeddings for Entity Ranking: A Replicability Study** submitted to ECIR 2023 Reproducibility Track. 

```Code``` contains the re-implementation code of the entity re-ranking framework of the original paper *Gerritse et al. (2020) Graph-Embedding Empowered Entity Retrieval* and the code of the neural fine-tuning model.

```Runs``` contains the run files generated for replicability and fine-tuning.


## Entity Re-ranking Framework

This framework replicates the work of *Gerritse et al. (2020) Graph-Embedding Empowered Entity Retrieval*. The re-implementation code is found at Code/entity_reranking/entity_rerank.py that determines the embedding score between the entities linked in query and candidate set of entities. The steps to reproduce the replication study are given below.

Install the requirements using the command.

``` 
pip install -r requirements.txt 
```


***Please note that when you run the script below, the embeddings would be downloaded for Wikipedia2Vec, ERNIE and E-BERT which would take ~60 GB of space***

``` 
bash downloads.sh
```

The script [downloads.sh](downloads.sh) downloads the embeddings, re-ranking data and rank-lips library and stores in the ```Data/embeddings``` folder, ```Data/data``` folder and the rank-lips library is stored at ```Data/ranklips``` folder.

Next we generate the entity rankings with the embedding score between the candidate set of entities and entities linked in queries using Wikipedia2Vec, ERNIE and E-BERT entity embeddings.

``` 
bash entity_reranking.sh 
```

The above script, [entity_reranking.sh](entity_reranking.sh) generates the output files ```Output/entity_reranking/$dataset/$embeddings/rerank.run``` in the ```Output/entity_reranking``` folder.

We perform the interpolation between the embedding scores and the baseline using rank-lips library with the below script. ***The Rank-lips library is only provided for Linux systems. All experiments have been conducted on a Debian system.***

``` 
bash train_entity_reranking.sh 
```

The [train_entity_reranking.sh](train_entity_reranking.sh) script generates the output files ```Output/entity_reranking/$dataset/$embeddings/l2r-run-test.run``` in the ```Output/entity_reranking``` folder. We optimize the Co-ordinate Ascent algorithm for MAP with random restarts of 5.

The run file ```l2r-run-test.run``` in the ```Output/entity_reranking/$dataset/$embeddings``` folder is the final entity ranking file.


## Neural Fine-tuning Model

Please install the requirements using the command given above if not done so already. We first download the training and test data for both the TREC-CAR and DBpediaV2 datasets using the script [neural_downloads.sh](neural_downloads.sh).

``` 
bash neural_downloads.sh 
```

To train the model, please run the following script of [train_model.sh](train_model.sh).

``` 
bash train_model.sh --batch-size 1000 --use-cuda --cuda $cuda
```

Please provide the cuda device numbers such 0 or 1 depending on your machine for ```$cuda```. This will train the pairwise and pointwise models for both the datasets. 


If you want to run the model on cpu please run the following command.

``` 
bash train_model.sh --batch-size 1000
```

The output will be stored in the folder ```Output/neural_model/$dataset/$embeddings``` folder. For TREC-CAR, the output of the test collection is stored in ```Output/neural_model/$dataset/$embeddings/$mode``` folder. The values of $mode parameter is either pointwise or pairwise. The final entity ranking of test collection is stored as ```Output/neural_model/$dataset/$embeddings/$mode/test.run```. For DBpediaV2, the output of the test is stored at the location ```Output/neural_model/$dataset``` folder. The final entity ranking of the test are stored as ```Output/neural_model/$dataset/$embeddings.$mode.test.run```.

The above script uses the batch size of 1000, 10 epochs, and learning rate of 2e-5.



