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

## Updated Results

***TREC CAR Y2 Test automatic results***

The following results are based on updated Baseline. Here we use a different ECM-based retrieval baseline with better performance. We discovered test-data leakage in the baseline used in the paper and we rectify the issue in this baseline alongwith a better retrieval performance.

| Model | MAP | P@R |
|-------|-----|-----|
|Wiki2Vec|0.1001|0.1489|
|ERNIE|0.0668|0.1045|
|E-BERT|0.0948|0.1429|
|Baseline|0.1606|0.2302|
|+Wiki2Vec|0.1686|0.2403|
|+ERNIE|0.1590|0.2326|
|+E-BERT|0.1663|0.2434|
|Wiki2Vec-Pairwise|0.5411|0.5276|
|Wiki2Vec-Pointwise|0.4351|0.4389|
|ERNIE-Pairwise|0.4866|0.4884|
|ERNIE-Pointwise|0.4358|0.4495|
|E-BERT-Pairwise|0.6375|0.6505|
|E-BERT-Pointwise|0.6109|0.6116|


***DBpediaV2 results***

| Dataset | DBpediaV2-All | INEX_LD | QALD-2 | SemSearch | ListSearch |
|---------|---------------|---------|--------|-----------|------------|
| Model | MAP | P@R |MAP | P@R |MAP | P@R |MAP | P@R |MAP | P@R |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|Wiki2Vec|0.3603|0.3824|0.3247|0.3466|0.3008|0.3150|0.4279|0.4333|0.3971|0.4453|
|ERNIE|0.2866|0.3251|0.2432|0.3001|0.2416|0.2607|0.3385|0.3548|0.3277|0.3959|
|E-BERT|0.3462|0.3713|0.3070|0.3394|0.2886|0.3025|0.4162|0.4341|0.3813|0.4206|
|Baseline|0.4536|0.4330|0.4195|0.4135|0.3657|0.3585|0.6058|0.5487|0.4406|0.4269|
|+Wiki2Vec|0.4540|0.4310|0.4132|0.4073|0.3705|0.3552|0.5945|0.5407|0.4528|0.4358|
|+ERNIE|0.4587|0.4356|0.4256|0.4169|0.3708|0.3636|0.6013|0.5380|0.4541|0.4386|
|+E-BERT|0.4551|0.4333|0.4226|0.4138|0.3667|0.3575|0.6012|0.5484|0.4472|0.4295|
|Wiki2Vec-Pairwise|0.5408|0.5512|0.5246|0.5491|0.5603|0.5461|0.5212|0.5488|0.5502|0.5616|
|Wiki2Vec-Pointwise|0.5049|0.5204|0.4852|0.5217|0.5281|0.5301|0.4869|0.5037|0.5115|0.5238|
|ERNIE-Pairwise|0.4910|0.5194|0.4543|0.5251|0.5196|0.5171|0.4656|0.4955|0.5127|0.5409|
|ERNIE-Pointwise|0.4851|0.5205|0.4607|0.5259|0.5281|0.5389|0.4234|0.4662|



