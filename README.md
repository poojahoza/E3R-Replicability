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

<table>
<thead>
<tr>
<th>Dataset  <td colspan=2> DBpediaV2-All  <td colspan=2> INEX_LD <td colspan=2> QALD-2 <td colspan=2> SemSearch <td colspan=2> ListSearch </td></th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Model</strong></td>
<td><strong>MAP</strong></td>
<td><strong>P@R</strong></td>
<td><strong>MAP</strong></td>
<td><strong>P@R</strong></td>
<td><strong>MAP</strong></td>
<td><strong>P@R</strong></td>
<td><strong>MAP</strong></td>
<td><strong>P@R</strong></td>
<td><strong>MAP</strong></td>
<td><strong>P@R</strong></td>
</tr>
<tr>
<td>Wiki2Vec</td>
<td>0.3603</td>
<td>0.3824</td>
<td>0.3247</td>
<td>0.3466</td>
<td>0.3008</td>
<td>0.3150</td>
<td>0.4279</td>
<td>0.4333</td>
<td>0.3971</td>
<td>0.4453</td>
</tr>
<tr>
<td>ERNIE</td>
<td>0.2866</td>
<td>0.3251</td>
<td>0.2432</td>
<td>0.3001</td>
<td>0.2416</td>
<td>0.2607</td>
<td>0.3385</td>
<td>0.3548</td>
<td>0.3277</td>
<td>0.3959</td>
</tr>
<tr>
<td>E-BERT</td>
<td>0.3462</td>
<td>0.3713</td>
<td>0.3070</td>
<td>0.3394</td>
<td>0.2886</td>
<td>0.3025</td>
<td>0.4162</td>
<td>0.4341</td>
<td>0.3813</td>
<td>0.4206</td>
</tr>
<tr>
<td>Baseline</td>
<td>0.4536</td>
<td>0.4330</td>
<td>0.4195</td>
<td>0.4135</td>
<td>0.3657</td>
<td>0.3585</td>
<td>0.6058</td>
<td>0.5487</td>
<td>0.4406</td>
<td>0.4269</td>
</tr>
<tr>
<td>+Wiki2Vec</td>
<td>0.4540</td>
<td>0.4310</td>
<td>0.4132</td>
<td>0.4073</td>
<td>0.3705</td>
<td>0.3552</td>
<td>0.5945</td>
<td>0.5407</td>
<td>0.4528</td>
<td>0.4358</td>
</tr>
<tr>
<td>+ERNIE</td>
<td>0.4587</td>
<td>0.4356</td>
<td>0.4256</td>
<td>0.4169</td>
<td>0.3708</td>
<td>0.3636</td>
<td>0.6013</td>
<td>0.5380</td>
<td>0.4541</td>
<td>0.4386</td>
</tr>
<tr>
<td>+E-BERT</td>
<td>0.4551</td>
<td>0.4333</td>
<td>0.4226</td>
<td>0.4138</td>
<td>0.3667</td>
<td>0.3575</td>
<td>0.6012</td>
<td>0.5484</td>
<td>0.4472</td>
<td>0.4295</td>
</tr>
<tr>
<td>Wiki2Vec-Pairwise</td>
<td>0.5408</td>
<td>0.5512</td>
<td>0.5246</td>
<td>0.5491</td>
<td>0.5603</td>
<td>0.5461</td>
<td>0.5212</td>
<td>0.5488</td>
<td>0.5502</td>
<td>0.5616</td>
</tr>
<tr>
<td>Wiki2Vec-Pointwise</td>
<td>0.5049</td>
<td>0.5204</td>
<td>0.4852</td>
<td>0.5217</td>
<td>0.5281</td>
<td>0.5301</td>
<td>0.4869</td>
<td>0.5037</td>
<td>0.5115</td>
<td>0.5238</td>
</tr>
<tr>
<td>ERNIE-Pairwise</td>
<td>0.4910</td>
<td>0.5194</td>
<td>0.4543</td>
<td>0.5251</td>
<td>0.5196</td>
<td>0.5171</td>
<td>0.4656</td>
<td>0.4955</td>
<td>0.5127</td>
<td>0.5409</td>
</tr>
<tr>
<td>ERNIE-Pointwise</td>
<td>0.4851</td>
<td>0.5205</td>
<td>0.4607</td>
<td>0.5259</td>
<td>0.5281</td>
<td>0.5389</td>
<td>0.4234</td>
<td>0.4662</td>
</tr>
</tbody>
</table>



