#!/bin/sh

dataset=TREC-CAR
embeddings=wiki2vec

mkdir -p Output/entity_reranking/$dataset/rank-lips-output/$embeddings


./Data/ranklips/rank-lips train -d Output/entity_reranking/$dataset/$embeddings/ -q Runs/Replicability/$dataset/benchmarkY2test-auto-article.entity.qrels -e l2r -O Output/entity_reranking/$dataset/rank-lips-output/$embeddings -o l2r --z-score --feature-variant FeatScore --mini-batch-size 1000 --convergence-threshold 0.001 --folds 5 --restarts 5 --threads 5 --train-cv --trec-eval-run

cp Output/entity_reranking/$dataset/rank-lips-output/$embeddings/l2r-run-test.run Output/entity_reranking/$dataset/$embeddings/


embeddings=ernie

mkdir -p Output/entity_reranking/$dataset/rank-lips-output/$embeddings


./Data/ranklips/rank-lips train -d Output/entity_reranking/$dataset/$embeddings/ -q Runs/Replicability/$dataset/benchmarkY2test-auto-article.entity.qrels -e l2r -O Output/entity_reranking/$dataset/rank-lips-output/$embeddings -o l2r --z-score --feature-variant FeatScore --mini-batch-size 1000 --convergence-threshold 0.001 --folds 5 --restarts 5 --threads 5 --train-cv --trec-eval-run

cp Output/entity_reranking/$dataset/rank-lips-output/$embeddings/l2r-run-test.run Output/entity_reranking/$dataset/$embeddings/


embeddings=ebert

mkdir -p Output/entity_reranking/$dataset/rank-lips-output/$embeddings

./Data/ranklips/rank-lips train -d Output/entity_reranking/$dataset/$embeddings/ -q Runs/Replicability/$dataset/benchmarkY2test-auto-article.entity.qrels -e l2r -O Output/entity_reranking/$dataset/rank-lips-output/$embeddings -o l2r --z-score --feature-variant FeatScore --mini-batch-size 1000 --convergence-threshold 0.001 --folds 5 --restarts 5 --threads 5 --train-cv --trec-eval-run

cp Output/entity_reranking/$dataset/rank-lips-output/$embeddings/l2r-run-test.run Output/entity_reranking/$dataset/$embeddings/

dataset=DBpediaV2
embeddings=wiki2vec

mkdir -p Output/entity_reranking/$dataset/rank-lips-output/$embeddings


./Data/ranklips/rank-lips train -d Output/entity_reranking/$dataset/$embeddings/ -q Runs/Replicability/$dataset/All/train.pages.cbor-article.entity.qrels -e l2r -O Output/entity_reranking/$dataset/rank-lips-output/$embeddings -o l2r --z-score --feature-variant FeatScore --mini-batch-size 1000 --convergence-threshold 0.001 --folds 5 --restarts 5 --threads 5 --train-cv --trec-eval-run

cp Output/entity_reranking/$dataset/rank-lips-output/$embeddings/l2r-run-test.run Output/entity_reranking/$dataset/$embeddings/

embeddings=ernie

mkdir -p Output/entity_reranking/$dataset/rank-lips-output/$embeddings


./Data/ranklips/rank-lips train -d Output/entity_reranking/$dataset/$embeddings/ -q Runs/Replicability/$dataset/All/train.pages.cbor-article.entity.qrels -e l2r -O Output/entity_reranking/$dataset/rank-lips-output/$embeddings -o l2r --z-score --feature-variant FeatScore --mini-batch-size 1000 --convergence-threshold 0.001 --folds 5 --restarts 5 --threads 5 --train-cv --trec-eval-run

cp Output/entity_reranking/$dataset/rank-lips-output/$embeddings/l2r-run-test.run Output/entity_reranking/$dataset/$embeddings/


embeddings=ebert

mkdir -p Output/entity_reranking/$dataset/rank-lips-output/$embeddings


./Data/ranklips/rank-lips train -d Output/entity_reranking/$dataset/$embeddings/ -q Runs/Replicability/$dataset/All/train.pages.cbor-article.entity.qrels -e l2r -O Output/entity_reranking/$dataset/rank-lips-output/$embeddings -o l2r --z-score --feature-variant FeatScore --mini-batch-size 1000 --convergence-threshold 0.001 --folds 5 --restarts 5 --threads 5 --train-cv --trec-eval-run

cp Output/entity_reranking/$dataset/rank-lips-output/$embeddings/l2r-run-test.run Output/entity_reranking/$dataset/$embeddings/


