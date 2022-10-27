#!/bin/sh

dataset=TREC-CAR
embeddings=wiki2vec

mkdir -p Output/entity_reranking/$dataset
mkdir -p Output/entity_reranking/$dataset/$embeddings

python Code/entity_reranking/entity_rerank.py --run Runs/Replicability/$dataset/baseline.run --annotations Data/data/$dataset/test/query_annotations.tsv --embeddings Data/embeddings/$embeddings/$embeddings.json --embedding-method $embeddings --name2id Data/name2id.tsv --k 1000 --save Output/entity_reranking/$dataset/$embeddings/rerank.run

cp Runs/Replicability/$dataset/baseline.run Output/entity_reranking/$dataset/$embeddings/


embeddings=ernie

mkdir -p Output/entity_reranking/$dataset/$embeddings

python Code/entity_reranking/entity_rerank.py --run Runs/Replicability/$dataset/baseline.run --annotations Data/data/$dataset/test/query_annotations.tsv --embeddings Data/embeddings/$embeddings/$embeddings.json --embedding-method $embeddings --name2id Data/name2id.tsv --k 1000 --save Output/entity_reranking/$dataset/$embeddings/rerank.run

cp Runs/Replicability/$dataset/baseline.run Output/entity_reranking/$dataset/$embeddings/


embeddings=ebert

mkdir -p Output/entity_reranking/$dataset/$embeddings

python Code/entity_reranking/entity_rerank.py --run Runs/Replicability/$dataset/baseline.run --annotations Data/data/$dataset/test/query_annotations.tsv --embeddings Data/embeddings/$embeddings/$embeddings.json --embedding-method $embeddings --name2id Data/name2id.tsv --k 1000 --save Output/entity_reranking/$dataset/$embeddings/rerank.run

cp Runs/Replicability/$dataset/baseline.run Output/entity_reranking/$dataset/$embeddings/


dataset=DBpediaV2
embeddings=wiki2vec

mkdir -p Output/entity_reranking/$dataset
mkdir -p Output/entity_reranking/$dataset/$embeddings


python Code/entity_reranking/entity_rerank.py --run Runs/Replicability/$dataset/All/baseline.run --annotations Data/data/$dataset/query_annotations.tsv --embeddings Data/embeddings/$embeddings/$embeddings.json --embedding-method $embeddings --name2id Data/name2id.tsv --k 1000 --save Output/entity_reranking/$dataset/$embeddings/rerank.run

cp Runs/Replicability/$dataset/All/baseline.run Output/entity_reranking/$dataset/$embeddings/

embeddings=ernie

mkdir -p Output/entity_reranking/$dataset/$embeddings

python Code/entity_reranking/entity_rerank.py --run Runs/Replicability/$dataset/All/baseline.run --annotations Data/data/$dataset/query_annotations.tsv --embeddings Data/embeddings/$embeddings/$embeddings.json --embedding-method $embeddings --name2id Data/name2id.tsv --k 1000 --save Output/entity_reranking/$dataset/$embeddings/rerank.run

cp Runs/Replicability/$dataset/All/baseline.run Output/entity_reranking/$dataset/$embeddings/


embeddings=ebert

mkdir -p Output/entity_reranking/$dataset/$embeddings

python Code/entity_reranking/entity_rerank.py --run Runs/Replicability/$dataset/All/baseline.run --annotations Data/data/$dataset/query_annotations.tsv --embeddings Data/embeddings/$embeddings/$embeddings.json --embedding-method $embeddings --name2id Data/name2id.tsv --k 1000 --save Output/entity_reranking/$dataset/$embeddings/rerank.run

cp Runs/Replicability/$dataset/All/baseline.run Output/entity_reranking/$dataset/$embeddings/

