mkdir -p Data/data/TREC-CAR/train

wget -P Data/data/TREC-CAR/train/ http://trec-car.cs.unh.edu/appendix/GEER-repro/data/TREC-CAR/train/set-0.tar.gz
wget -P Data/data/TREC-CAR/train/ http://trec-car.cs.unh.edu/appendix/GEER-repro/data/TREC-CAR/train/set-1.tar.gz
wget -P Data/data/TREC-CAR/train/ http://trec-car.cs.unh.edu/appendix/GEER-repro/data/TREC-CAR/train/set-2.tar.gz
wget -P Data/data/TREC-CAR/train/ http://trec-car.cs.unh.edu/appendix/GEER-repro/data/TREC-CAR/train/set-3.tar.gz
wget -P Data/data/TREC-CAR/train/ http://trec-car.cs.unh.edu/appendix/GEER-repro/data/TREC-CAR/train/set-4.tar.gz
wget -P Data/data/TREC-CAR/train/ http://trec-car.cs.unh.edu/appendix/GEER-repro/data/TREC-CAR/train/query_annotations.tsv
wget -P Data/data/TREC-CAR/train/ http://trec-car.cs.unh.edu/appendix/GEER-repro/data/TREC-CAR/train/train/ernie.tar.gz
wget -P Data/data/TREC-CAR/train/ http://trec-car.cs.unh.edu/appendix/GEER-repro/data/TREC-CAR/train/train/wiki2vec.tar.gz

tar -xvzf Data/data/TREC-CAR/train/set-0.tar.gz -C Data/data/TREC-CAR/train && rm -rf Data/data/TREC-CAR/train/set-0.tar.gz
tar -xvzf Data/data/TREC-CAR/train/set-1.tar.gz -C Data/data/TREC-CAR/train && rm -rf Data/data/TREC-CAR/train/set-1.tar.gz
tar -xvzf Data/data/TREC-CAR/train/set-2.tar.gz -C Data/data/TREC-CAR/train && rm -rf Data/data/TREC-CAR/train/set-2.tar.gz
tar -xvzf Data/data/TREC-CAR/train/set-3.tar.gz -C Data/data/TREC-CAR/train && rm -rf Data/data/TREC-CAR/train/set-3.tar.gz
tar -xvzf Data/data/TREC-CAR/train/set-4.tar.gz -C Data/data/TREC-CAR/train && rm -rf Data/data/TREC-CAR/train/set-4.tar.gz
tar -xvzf Data/data/TREC-CAR/train/ernie.tar.gz -C Data/data/TREC-CAR/train && rm -rf Data/data/TREC-CAR/train/ernie.tar.gz
tar -xvzf Data/data/TREC-CAR/train/wiki2vec.tar.gz -C Data/data/TREC-CAR/train && rm -rf Data/data/TREC-CAR/train/wiki2vec.tar.gz

mkdir -p Data/data/TREC-CAR/test/wiki2vec
mkdir -p Data/data/TREC-CAR/test/ernie
mkdir -p Data/data/TREC-CAR/test/ebert

wget -P Data/data/TREC-CAR/test/wiki2vec http://trec-car.cs.unh.edu/appendix/GEER-repro/data/TREC-CAR/test/wiki2vec/test.jsonl
wget -P Data/data/TREC-CAR/test/ernie http://trec-car.cs.unh.edu/appendix/GEER-repro/data/TREC-CAR/test/ernie/test.jsonl
wget -P Data/data/TREC-CAR/test/ebert http://trec-car.cs.unh.edu/appendix/GEER-repro/data/TREC-CAR/test/ebert/test.jsonl


mkdir -p Data/data/DBpediaV2/train

wget -P Data/data/DBpediaV2/train http://trec-car.cs.unh.edu/appendix/GEER-repro/data/DBpediaV2/train/set-0.tar.gz
wget -P Data/data/DBpediaV2/train http://trec-car.cs.unh.edu/appendix/GEER-repro/data/DBpediaV2/train/set-1.tar.gz
wget -P Data/data/DBpediaV2/train http://trec-car.cs.unh.edu/appendix/GEER-repro/data/DBpediaV2/train/set-2.tar.gz
wget -P Data/data/DBpediaV2/train http://trec-car.cs.unh.edu/appendix/GEER-repro/data/DBpediaV2/train/set-3.tar.gz
wget -P Data/data/DBpediaV2/train http://trec-car.cs.unh.edu/appendix/GEER-repro/data/DBpediaV2/train/set-4.tar.gz

tar -xvzf Data/data/DBpediaV2/train/set-0.tar.gz -C Data/data/DBpediaV2/train && rm -rf Data/data/DBpediaV2/train/set-0.tar.gz
tar -xvzf Data/data/DBpediaV2/train/set-1.tar.gz -C Data/data/DBpediaV2/train && rm -rf Data/data/DBpediaV2/train/set-1.tar.gz
tar -xvzf Data/data/DBpediaV2/train/set-2.tar.gz -C Data/data/DBpediaV2/train && rm -rf Data/data/DBpediaV2/train/set-2.tar.gz
tar -xvzf Data/data/DBpediaV2/train/set-3.tar.gz -C Data/data/DBpediaV2/train && rm -rf Data/data/DBpediaV2/train/set-3.tar.gz
tar -xvzf Data/data/DBpediaV2/train/set-4.tar.gz -C Data/data/DBpediaV2/train && rm -rf Data/data/DBpediaV2/train/set-4.tar.gz

