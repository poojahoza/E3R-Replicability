mkdir -p Data/embeddings/wiki2vec
mkdir -p Data/embeddings/ernie
mkdir -p Data/embeddings/ebert
mkdir -p Data/ranklips
mkdir -p Data/data/TREC-CAR/test
mkdir -p Data/data/DBpediaV2

echo "Downloading embeddings..will take a while, why not take a break and get a cup of coffee?"

wget -P Data/embeddings/wiki2vec http://trec-car.cs.unh.edu/appendix/GEER-repro/embeddings/wiki2vec/wiki2vec.json.gz
wget -P Data/embeddings/ernie http://trec-car.cs.unh.edu/appendix/GEER-repro/embeddings/ernie/ernie.json.gz
wget -P Data/embeddings/ebert http://trec-car.cs.unh.edu/appendix/GEER-repro/embeddings/ebert/ebert.json.gz

echo "Done downloading embeddings"

echo "unzipping embeddings...this will take some time!"
gzip -d Data/embeddings/wiki2vec/wiki2vec.json.gz

echo "unzipped wiki2vec embeddings"

gzip -d Data/embeddings/ernie/ernie.json.gz

echo "unzipped ernie embeddings"


gzip -d Data/embeddings/ebert/ebert.json.gz

echo "unzipped ebert embeddings"

echo "Completed embeddings processing"

wget -P Data/data/TREC-CAR/test http://trec-car.cs.unh.edu/appendix/GEER-repro/data/TREC-CAR/test/query_annotations.tsv
wget -P Data/data/DBpediaV2 http://trec-car.cs.unh.edu/appendix/GEER-repro/data/DBpediaV2/query_annotations.tsv

wget -P Data/ http://trec-car.cs.unh.edu/appendix/GEER-repro/name2id.tsv
wget -P Data/ http://trec-car.cs.unh.edu/appendix/GEER-repro/id2name.tsv


wget -P Data/ranklips https://www.cs.unh.edu/~dietz/rank-lips/rank-lips
chmod +x Data/ranklips/rank-lips
