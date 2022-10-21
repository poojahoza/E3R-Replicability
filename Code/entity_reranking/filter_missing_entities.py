import os
import numpy as np
from typing import Dict, List
import argparse
import sys
import json
import tqdm
from scipy import spatial
import operator


def load_run_file(file_path: str) -> Dict[str, List[str]]:
    rankings:List[str] = []
    with open(file_path, 'r') as file:
        rankings = file.readlines()
    return rankings

def get_query_annotations(query_annotations: str) -> Dict[str, float]:
    annotations = json.loads(query_annotations)
    res: Dict[str, float] = {}
    for ann in annotations:
        a = json.loads(ann)
        res[a['entity_name']] = a['score']
    return res

def check_entity(
        query_entity: str,
        target_entity: str,
        embeddings: Dict[str, List[str]],
        name2id: Dict[str, str]
) -> int:
    if query_entity in name2id and name2id[query_entity].strip() in embeddings and target_entity in embeddings:
        return 1
    else:
        return 0

def filter_run_file(
        rankings: List[str],
        query_annotations_dict: Dict[str, str],
        embeddings: Dict[str, List[float]],
        name2id: Dict[str, str],
        out_file: str
) -> None:
    
    filtered_rankings: List[str] = []
    for ranking_item in tqdm.tqdm(rankings,total=len(rankings)):
        #print(ranking_item)
        item_split = ranking_item.split()
        #print(item_split[0].strip())
        #print(query_annotations.keys())
        #print(query_annotations_dict[item_split[0].strip()])
        query_annotations = get_query_annotations(query_annotations_dict[item_split[0]])
        target_entity = item_split[2]
        flag_absent_entity = 0
        for query_entity, conf in query_annotations.items():
            entities_flag = check_entity(query_entity, target_entity, embeddings, name2id)
            if entities_flag == 0:
                flag_absent_entity = 1
        if flag_absent_entity == 0:
            filtered_rankings.append(ranking_item)

    write_to_file(filtered_rankings, out_file)


def write_to_file(run_file_strings: List[str], run_file: str) -> None:
    with open(run_file, 'a') as f:
        for item in run_file_strings:
            f.write("%s" % item)


def read_tsv(file: str) -> Dict[str, str]:
    res = {}
    with open(file, 'r') as f:
        for line in f:
            parts = line.split('\t')
            key = parts[0]
            value = parts[1]
            res[key] = value
    return res


def main():
    """
    Main method to run code.
    """
    parser = argparse.ArgumentParser("Remove entities from the run file for which the embeddings are not present")
    parser.add_argument("--run", help="Entity run file to filter", required=True)
    parser.add_argument("--annotations", help="File containing TagMe annotations for queries.", required=True)
    parser.add_argument("--embeddings", help="Entity embedding file", required=True)
    parser.add_argument("--name2id", help="EntityName to EntityId mappings.", required=True)
    parser.add_argument("--save", help="Output run file (re-ranked).", required=True)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading run file...')
    rankings_list: List[str] = load_run_file(args.run)
    print('[Done].')

    print('Loading query annotations...')
    query_annotations: Dict[str, str] = read_tsv(args.annotations)
    print('[Done].')

    print('Loading entity embeddings...')
    with open(args.embeddings, 'r') as f:
        embeddings: Dict[str, List[float]] = json.load(f)
    print('[Done].')

    print('Loading name2id file...')
    name2id = read_tsv(args.name2id)
    print('[Done].')

    print("Filtering run...")
    filter_run_file(
        rankings=rankings_list,
        query_annotations_dict=query_annotations,
        embeddings=embeddings,
        name2id=name2id,
        out_file=args.save
    )
    print('[Done].')

    print('New run file written to {}'.format(args.save))


if __name__ == '__main__':
    main()
