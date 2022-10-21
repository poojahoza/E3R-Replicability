import os
import numpy as np
from typing import Dict, List
import argparse
import sys
import json
import tqdm
from scipy import spatial
import operator

missing_candidate_entities = set()
missing_query_entities = set()
missing_entity_in_name2id = set()
total_candidate_entities = set()
total_query_entities = set()
missing_relevant_candidate_entities = set()


def load_qrel_file(qrel_file_path: str) -> Dict[str, List[str]]:
    qrels: Dict[str, List[str]] = {}
    with open(qrel_file_path, 'r') as qrel_file:
        for line in qrel_file:
            line_split = line.split(" ")
            query_id = line_split[0]
            candidate_entity = line_split[2]
            relevance = int(line_split[3])
            entity_list: List[str] = qrels[query_id][0] if query_id in qrels.keys() else []
            relevance_list: List[int] = qrels[query_id][1] if query_id in qrels.keys() else []
            entity_list.append(candidate_entity)
            relevance_list.append(relevance)
            qrels[query_id] = [entity_list, relevance_list]
    return qrels


def load_run_file(file_path: str) -> Dict[str, List[str]]:
    rankings: Dict[str, List[str]] = {}
    with open(file_path, 'r') as file:
        for line in file:
            line_parts = line.split(" ")
            query_id = line_parts[0]
            entity_id = line_parts[2]
            entity_list: List[str] = rankings[query_id] if query_id in rankings.keys() else []
            entity_list.append(entity_id)
            rankings[query_id] = entity_list
    return rankings

def get_query_annotations(query_annotations: str) -> Dict[str, float]:
    annotations = json.loads(query_annotations)
    res: Dict[str, float] = {}
    for ann in annotations:
        a = json.loads(ann)
        res[a['entity_name']] = a['score']
    return res

def get_missing_entities(
        query_entity: str,
        target_entity: str,
        embeddings: Dict[str, List[float]],
        name2id: Dict[str, str],
        qrels: List[str],
        rels: List[int],
) -> None:
    total_candidate_entities.add(target_entity)
    if query_entity in name2id:
        total_query_entities.add(query_entity)
    if query_entity in name2id and name2id[query_entity].strip() not in embeddings:
        missing_query_entities.add(query_entity)
    if target_entity not in embeddings:
        missing_candidate_entities.add(target_entity)
    if query_entity not in name2id:
        #print("query missing in name2id", query_entity)
        missing_entity_in_name2id.add(query_entity)
    if target_entity not in embeddings and target_entity in qrels:
        rel_index = qrels.index(target_entity)
        if rels[rel_index] == 1:
            missing_relevant_candidate_entities.add(target_entity)


def check_entities(
        run_dict: Dict[str, List[str]],
        query_annotations: Dict[str, str],
        embeddings: Dict[str, List[float]],
        embedding_method: str,
        name2id: Dict[str, str],
        qrels: Dict[str, List[str]],
        out_file: str
) -> None:

    for query_id, query_entities in tqdm.tqdm(run_dict.items(), total=len(run_dict)):
        qrel_entities = qrels[query_id]
        for entity in query_entities:
            for query_entity, query_conf_score in get_query_annotations(query_annotations[query_id]).items():
                #print(query_id, query_entity, entity)
                get_missing_entities(query_entity, entity, embeddings, name2id, qrel_entities[0], qrel_entities[1])
    final_output = []
    final_output.append("Missing candidate entities in embeddings : "+str(len(missing_candidate_entities)))
    final_output.append("Total candidate entities : "+str(len(total_candidate_entities)))
    final_output.append("Missing query entities in embeddings : "+str(len(missing_query_entities)))
    final_output.append("Total query entities linked : "+str(len(total_query_entities)))
    final_output.append("Missing entities in name2id : "+str(len(missing_entity_in_name2id)))
    final_output.append("Missing relevant candidate entities in embeddings : "+str(len(missing_relevant_candidate_entities)))
    print("Missing candidate entities in embeddings : ",len(missing_candidate_entities))
    print("Missing query entities in embeddings : ",len(missing_query_entities))
    print("Missing entities in name2id : ",len(missing_entity_in_name2id))
    print("Total candidate entities : ",len(total_candidate_entities))
    print("Total entities linked in queries : ",len(total_query_entities))
    print("Missing relevant candidate entities in embeddings : ",len(missing_relevant_candidate_entities))

    write_to_file(final_output, out_file)


def write_to_file(run_file_strings: List[str], run_file: str) -> None:
    with open(run_file, 'a') as f:
        for item in run_file_strings:
            f.write("%s\n" % item)


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
    parser = argparse.ArgumentParser("Re-implementation of entity re-ranking using entity embeddings from Gerritse et al., 2020.")
    parser.add_argument("--run", help="Entity run file to re-rank.", required=True)
    parser.add_argument("--annotations", help="File containing TagMe annotations for queries.", required=True)
    parser.add_argument("--qrels", help="File containing qrels", required=True)
    parser.add_argument("--embeddings", help="Entity embedding file", required=True)
    parser.add_argument("--embedding-method", help="Entity embedding method (Wiki2Vec|ERNIE|E-BERT).", required=True)
    parser.add_argument("--name2id", help="EntityName to EntityId mappings.", required=True)
    parser.add_argument("--k", help="Top-K entities to re-rank from run file.", required=True, type=int)
    parser.add_argument("--save", help="Output run file (re-ranked).", required=True)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading run file...')
    run_dict: Dict[str, List[str]] = load_run_file(args.run)
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
    print('[Donei].')

    print('Loading qrel file...')
    qrels: Dict[str, List[str]] = load_qrel_file(args.qrels)

    print("Getting missing entities....")
    check_entities(
            run_dict=run_dict,
            query_annotations=query_annotations,
            embeddings=embeddings,
            embedding_method=args.embedding_method,
            name2id=name2id,
            qrels=qrels,
            out_file=args.save
    )

    print('[Done].')

    print('New run file written to {}'.format(args.save))


if __name__ == '__main__':
    main()
