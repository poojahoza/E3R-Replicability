#!/bin/bash

usage() {
	echo "A simple bash script to automate 5-fold CV."
	echo "usage: ./train_cv.sh [--use-cuda USE CUDA] [--batch-size BATCH SIZE] [--folds FOLDS]"
	echo "    --use-cuda      USE CUDA        Whether to use CUDA or not (true|false)."
	echo "    --batch-size    BATCH SIZE      Size of each batch during training."
	echo "    --folds         FOLDS           Folds to consider (Type one after another like 0 1 2 ...). Defaults to all folds if absent."
}

if [ "$#" -eq 0 ]; then
   	usage
	  exit 1
fi
# Get the command line parameters

while [ "$1" != "" ];
do
	    case $1 in

		--use-cuda         )            shift
		                        	      useCuda=$1
		                        	      ;;
		--batch-size       )            shift
    		                        	  batchSize=$1
    		                        	  ;;

    --folds            )            shift
        						                folds=( "$@" )
        						                ;;

		-h | --help        )           usage
		                               exit
		                               ;;


	    esac
	    shift
done

if [ ${#folds[@]} -eq 0 ]; then
    echo "Folds not specified. Using all folds."
    folds=( "0" "1" "2" "3" "4" )
fi

# Arguments
trainScript="Code/neural_model/train.py"
dataDir="Data/data/DBpediaV2/train"
outDir="Output/neural_model/DBpediaV2"
epoch="10"
CUDA_VISIBLE_DEVICES=0,1

echo "Batch Size = $batchSize"
echo "Epochs = $epoch"

for embeddingName in "wiki2vec" "ernie"; do
  echo "==================================="
  echo "Embedding: $embeddingName"
  for mode in "pairwise" "pointwise"; do
    echo "Mode: $mode"
    trainFile="train.$mode.jsonl"
    echo
    for i in "${folds[@]}"; do
      echo "Set: $i"
      trainData=$dataDir/"set-"$i/$embeddingName/$trainFile
      devData=$dataDir/"set-"$i/$embeddingName/"test.jsonl"
      devQrels=$dataDir/"set-"$i/"test.qrels"
      mkdir -p $outDir/"set-"$i/$embeddingName/$mode
      savePath=$outDir/"set-"$i/$embeddingName/$mode
      echo "Loading train data from ==> $trainData"
      echo "Loading dev data from ==> $devData"
      echo "Saving results to: $savePath"
      if [[ "${useCuda}" == "true" ]]; then
            python3 $trainScript --model-type $mode --train "$trainData" --dev "$devData" --qrels "$devQrels" --save-dir "$savePath" --run "test.run" --epoch $epoch --in-emb-dim 100 --out-emb-dim 100 --batch-size "$batchSize" --use-cuda --cuda 1
        else
            python3 $trainScript --model-type $mode --train "$trainData" --dev "$devData" --qrels "$devQrels" --save-dir "$savePath" --run "test.run" --epoch $epoch --in-emb-dim 100 --out-emb-dim 100 --batch-size "$batchSize"
      fi
      echo
    done
  done
done

cat $outDir/set-0/wiki2vec/pairwise/test.run $outDir/set-1/wiki2vec/pairwise/test.run $outDir/set-2/wiki2vec/pairwise/test.run $outDir/set-3/wiki2vec/pairwise/test.run $outDir/set-4/wiki2vec/pairwise/test.run > $outDir/wiki2vec.pairwise.test.run

cat $outDir/set-0/wiki2vec/pointwise/test.run $outDir/set-1/wiki2vec/pointwise/test.run $outDir/set-2/wiki2vec/pointwise/test.run $outDir/set-3/wiki2vec/pointwise/test.run $outDir/set-4/wiki2vec/pointwise/test.run > $outDir/wiki2vec.pointwise.test.run

cat $outDir/set-0/ernie/pairwise/test.run $outDir/set-1/ernie/pairwise/test.run $outDir/set-2/ernie/pairwise/test.run $outDir/set-3/ernie/pairwise/test.run $outDir/set-4/ernie/pairwise/test.run > $outDir/ernie.pairwise.test.run

cat $outDir/set-0/ernie/pointwise/test.run $outDir/set-1/ernie/pointwise/test.run $outDir/set-2/ernie/pointwise/test.run $outDir/set-3/ernie/pointwise/test.run $outDir/set-4/ernie/pointwise/test.run > $outDir/ernie.pointwise.test.run


echo "Training continues..."


dataDir="Data/data/TREC-CAR/train/wiki2vec"
mkdir -p Output/neural_model/TREC-CAR/wiki2vec/pairwise
outDir="Output/neural_model/TREC-CAR/wiki2vec/pairwise"
trainData=$dataDir/"train.pairwise.jsonl"
devData="Data/data/TREC-CAR/test/wiki2vec/test.jsonl"
devQrels="Runs/Replicability/TREC-CAR/benchmarkY2test-auto-article.entity.qrels"
savePath=$outDir
mode="pairwise"

if [[ "${useCude}" == "true" ]]; then
	python3 $trainScript --model-type $mode --train "$trainData" --dev "$devData" --qrels "$devQrels" --save-dir "$savePath" --run "test.run" --epoch $epoch --in-emb-dim 100 --out-emb-dim 100 --batch-size "$batchSize" --use-cuda --cuda 1
else
	python3 $trainScript --model-type $mode --train "$trainData" --dev "$devData" --qrels "$devQrels" --save-dir "$savePath" --run "test.run" --epoch $epoch --in-emb-dim 100 --out-emb-dim 100 --batch-size "$batchSize"
fi


dataDir="Data/data/TREC-CAR/train/wiki2vec"
mkdir -p Output/neural_model/TREC-CAR/wiki2vec/pointwise
outDir="Output/neural_model/TREC-CAR/wiki2vec/pointwise"
trainData=$dataDir/"train.pointwise.jsonl"
devData="Data/data/TREC-CAR/test/wiki2vec/test.jsonl"
devQrels="Runs/Replicability/TREC-CAR/benchmarkY2test-auto-article.entity.qrels"
savePath=$outDir
mode="pointwise"

if [[ "${useCude}" == "true" ]]; then
	python3 $trainScript --model-type $mode --train "$trainData" --dev "$devData" --qrels "$devQrels" --save-dir "$savePath" --run "test.run" --epoch $epoch --in-emb-dim 100 --out-emb-dim 100 --batch-size "$batchSize" --use-cuda --cuda 1
else
	python3 $trainScript --model-type $mode --train "$trainData" --dev "$devData" --qrels "$devQrels" --save-dir "$savePath" --run "test.run" --epoch $epoch --in-emb-dim 100 --out-emb-dim 100 --batch-size "$batchSize"
fi



dataDir="Data/data/TREC-CAR/train/ernie"
mkdir -p Output/neural_model/TREC-CAR/ernie/pairwise
outDir="Output/neural_model/TREC-CAR/ernie/pairwise"
trainData=$dataDir/"train.pairwise.jsonl"
devData="Data/data/TREC-CAR/test/ernie/test.jsonl"
devQrels="Runs/Replicability/TREC-CAR/benchmarkY2test-auto-article.entity.qrels"
savePath=$outDir
mode="pairwise"

if [[ "${useCude}" == "true" ]]; then
	python3 $trainScript --model-type $mode --train "$trainData" --dev "$devData" --qrels "$devQrels" --save-dir "$savePath" --run "test.run" --epoch $epoch --in-emb-dim 100 --out-emb-dim 100 --batch-size "$batchSize" --use-cuda --cuda 1
else
	python3 $trainScript --model-type $mode --train "$trainData" --dev "$devData" --qrels "$devQrels" --save-dir "$savePath" --run "test.run" --epoch $epoch --in-emb-dim 100 --out-emb-dim 100 --batch-size "$batchSize"
fi


dataDir="Data/data/TREC-CAR/train/ernie"
mkdir -p Output/neural_model/TREC-CAR/ernie/pointwise
outDir="Output/neural_model/TREC-CAR/ernie/pointwise"
trainData=$dataDir/"train.pointwise.jsonl"
devData="Data/data/TREC-CAR/test/ernie/test.jsonl"
devQrels="Runs/Replicability/TREC-CAR/benchmarkY2test-auto-article.entity.qrels"
savePath=$outDir
mode="pointwise"

if [[ "${useCude}" == "true" ]]; then
	python3 $trainScript --model-type $mode --train "$trainData" --dev "$devData" --qrels "$devQrels" --save-dir "$savePath" --run "test.run" --epoch $epoch --in-emb-dim 100 --out-emb-dim 100 --batch-size "$batchSize" --use-cuda --cuda 1
else
	python3 $trainScript --model-type $mode --train "$trainData" --dev "$devData" --qrels "$devQrels" --save-dir "$savePath" --run "test.run" --epoch $epoch --in-emb-dim 100 --out-emb-dim 100 --batch-size "$batchSize"
fi





