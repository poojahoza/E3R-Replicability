import os
import numpy as np
from typing import Dict, List
import argparse
import sys
import json
import tqdm
from scipy import spatial
import operator


def load_run_file(file_path):
    final_output = dict()
    with open(file_path, 'r') as reader:
        for line in reader:
            data = line.split()
            queryid = data[0]
            entid = data[2]
            ent_list = []
            if queryid in final_output:
                ent_list = final_output[queryid]
            ent_list.append(entid)
            final_output[queryid] = ent_list
    return final_output

def read_annotations_file(input_file):
    final_output = dict()
    with open(input_file, 'r') as reader:
        for item in reader:
            data = item.split('\t')
            q1 = data[0]
            q2 = data[1]
            final_output[q1] = q2
    return final_output

def load_embeddings_file(embeddings_file):
    final_embeddings = dict()
    with open(embeddings_file, 'r') as reader:
        final_embeddings = json.load(reader)
    return final_embeddings

def calculate_score(embeddings,
        ent_1,
        ent_2):
    if ent_1 in embeddings and ent_2 in embeddings:
        embed_ent_1 = np.array(embeddings[ent_1])
        embed_ent_2 = np.array(embeddings[ent_2])

        return 1 - spatial.distance.cosine(embed_ent_1, embed_ent_2)
    else:
        return 0.0


def entity_reranking(run_data,
        query_ann,
        embed,
        name2id,
        top_k):

    query_ent_score = dict()
    
    for queryid, ent in tqdm.tqdm(run_data.items(), total=len(run_data)):

        top_k_ent = ent[:top_k]
        q_ann = query_ann[queryid]
        result_output = dict()

        ann = json.loads(q_ann)
        ann_dict = dict()
        for a in ann:
            data = json.loads(a)
            ann_dict[data['entity_name']] = data['score']

        ent_score_dict = dict()

        for e in ent:
            ent_score = 0.0
            for qent, cscore in ann_dict.items():
                cosine = 0.0
                if qent in name2id:
                    query_ent = name2id[qent].strip()
                    cosine = calculate_score(embed, query_ent, e)
                ent_score += cosine*cscore
            ent_score_dict[e] = ent_score

        query_ent_score[queryid] = ent_score_dict

    return query_ent_score



def write_to_txt_file(data,
        output_file):
    with open(output_file, 'w') as writer:
        for item in data:
            writer.write(item+"\n")


def reranking(run_data,
        query_ann,
        embed,
        name2id,
        top_k,
        method,
        output_file):

    query_reranking = entity_reranking(run_data,
            query_ann,
            embed,
            name2id,
            top_k)

    final_data = []

    for q,entities in tqdm.tqdm(query_reranking.items(), total=len(query_reranking)):
        sorted_entities = dict(sorted(entities.items(), key=operator.itemgetter(1), reverse=True))
        rank = 1
        for ent,score in sorted_entities.items():
            if score > 0.0:
                result_str = q+' Q0 '+ent+' '+str(rank)+' '+str(score)+' '+method+'-ReRank'
                final_data.append(result_str)
                rank += 1

    write_to_txt_file(final_data, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Re-implementation of entity re-ranking using entity embeddings from Gerritse et al., 2020.")
    parser.add_argument("--run", help="Entity run file to re-rank.", required=True)
    parser.add_argument("--annotations", help="File containing TagMe annotations for queries.", required=True)
    parser.add_argument("--embeddings", help="Entity embedding file", required=True)
    parser.add_argument("--embedding-method", help="Entity embedding method (Wiki2Vec|ERNIE|E-BERT).", required=True)
    parser.add_argument("--name2id", help="EntityName to EntityId mappings.", required=True)
    parser.add_argument("--k", help="Top-K entities to re-rank from run file.", required=True, type=int)
    parser.add_argument("--save", help="Output run file (re-ranked).", required=True)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading run file...')
    run_dict = load_run_file(args.run)
    print('[Done].')

    print('Loading query annotations...')
    query_annotations = read_annotations_file(args.annotations)
    print('[Done].')

    print('Loading entity embeddings...')
    embeddings = load_embeddings_file(args.embeddings)
    print('[Done].')

    print('Loading name2id file...')
    name2id = read_annotations_file(args.name2id)
    print('[Done].')

    print("Re-Ranking run...")
    reranking(
        run_dict,
        query_annotations,
        embeddings,
        name2id,
        args.k,
        args.embedding_method,
        args.save
    )
    print('[Done].')

    print('New run file written to {}'.format(args.save))

