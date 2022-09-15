"""
Utility functions to get the data from the file system
"""

import csv
import os
import pickle
import re
from typing import Dict, List, Tuple

import numpy as np

from concept_processing.extraction import ConceptExtractor, ConceptBag, ConceptsState
from concept_processing.nlp.nlp_parser import NLPParser
from concept_processing.pam import count_datapoints_in_each_feature


def get_participant_iterator(block, fragile=True):
    """
    Given a single participant as a list of lines. 
    Iterate over and yield one id, label and text per row.
    """
    for i, line in enumerate(block):
        if i == 0:
            head_tokens = line.split(',')
            continue
        try:
            line = line.strip()
            line_tokens = line.split(',')
            id_ = str(line_tokens[0])
            label = line_tokens[1]
            text = ','.join(line_tokens[2:]).strip()
            yield id_, label, text
        except:
            # print(f"Fails with line: {line}")
            if fragile:
                raise


def get_file_iterator(path, max_items=np.inf, ext=".csv", verbose=True):
    """
    For a given directory search over all files with extension ext and 
    yield one readable file object per iteration
    """
    count = 0
    for entry in os.scandir(path):
        if count >= max_items:
            break
        if entry.name.endswith(ext) and entry.is_file():
            # print(entry.name, entry.path)
            if verbose:
                print(entry.name, end=', ')
            with open(entry.path, 'r') as ifile:
                original = ifile.readlines()
                # print(f"original:\n{original}")
                filtered = filter(lambda x: not re.match(r'^\s*$', x), original)
                # print(f"filtered:\n{list(filtered)}")
                yield entry.path, filtered
                count += 1


def get_datapoint_iterator(
        path, max_files=np.inf, fragile=True, include_fname=False, verbose=True,
        rows_to_remove=None):
    """
    For a given directory iterate over all rows in all files with extension
    csv and yield one datapoint per iteration
    """
    i = 0
    for fname, block in get_file_iterator(path, max_items=max_files, ext='.csv', verbose=verbose):
        for datapoint in get_participant_iterator(block, fragile=fragile):
            if rows_to_remove is None or not rows_to_remove[i]:
                if include_fname:
                    yield (fname,) + datapoint
                else:
                    yield datapoint
            i = i + 1


def capture_all_concepts(nlp: NLPParser, examples_dir: str) -> Tuple[ConceptBag, List[str]]:
    concept_extractor = ConceptExtractor(nlp)

    for id_, label, text in get_datapoint_iterator(examples_dir):
        try:
            # we remove double whitespace as it breaks the benepar plugin
            text = text.replace('�', '')
            text = re.sub(' +', ' ', text)
        except Exception:
            print(f"Error occurred because of text = {text}")
            raise

        concept_extractor.parse(id_, text, label)

    return concept_extractor.get()


def capture_concepts_from_dataframe(
        nlp, df, id_field, text_field, label_field=None):
    """
    Iterate over all files over each contribution line in files.
    Convert to concepts and index these in concept dictionary.
    Store text as collection of concept ids.
    Returns concept_dict (text to id mapping) and rawbagofconcepts (list of lists of concept ids).
    """
    if label_field is None:
        df['dummy_label'] = 0
        label_field = 'dummy_label'
    concept_extractor = ConceptExtractor(nlp)
    for index, row in df.iterrows():
        id_ = row[id_field]
        label = row[label_field]
        text = row[text_field]
        try:
            # we remove double whitespace as it breaks the benepar plugin
            text = text.replace('�', '')
            text = ' '.join(text.split('\n'))
            text = re.sub(' +', ' ', text)
        except:
            print(f"text = {text}")
            print(f"type(text) = {type(text)}")
            raise
        concept_extractor.parse(id_, text, label)

    return concept_extractor.get()


def lookup_explanations(lookup_id: str, examples_dir: str) -> str:
    for id_, label, text in get_datapoint_iterator(examples_dir, verbose=False):
        if id_ == lookup_id:
            return text

    return "Not found"


def load_concept_examples(fname):
    premise_texts = []
    concepts_texts = []
    with open(fname, 'r') as ifile:
        reader = csv.reader(ifile, delimiter=',', quotechar='"', skipinitialspace=True)
        for premise_text, concepts_text in reader:
            premise_texts.append(premise_text)
            concepts_texts.append(concepts_text)
    return premise_texts, concepts_texts


### deprecated ###
def get_block_iterator(fname):
    """
    For multiple participants in the same file. Deprecated.
    """
    with open(fname, 'r') as ifile:
        block = []
        inblock = True
        for line in ifile:
            line = line.strip()
            if line != '':
                block.append(line)
                inblock = True
            elif inblock:
                yield block
                inblock = False
                block = []


def get_test_data_iterator(fname='sample_docs.txt'):
    # just runs the iterators for sanity
    for i, block in enumerate(get_block_iterator(fname)):
        print(f"participant  {i}")
        for datapoint in get_participant_iterator(block):
            yield datapoint


def store_concept_objects(path, concepts, ids, categories, labels, pam):
    pickle.dump((concepts, ids, categories, labels, pam), open(path, "wb"))


def load_concept_objects(path):
    concepts, ids, categories, labels, pam = pickle.load(open(path, "rb"))
    return concepts, ids, categories, labels, pam


def store_pruned_results(
        path, concepts, ids, categories, labels, pruned_grouped_concept_ids, pruned_grouped_pam):
    to_store = (concepts, ids, categories, labels, pruned_grouped_concept_ids, pruned_grouped_pam)
    pickle.dump(to_store, open(path, "wb"))


def load_pruned_results(path):
    concepts, ids, categories, labels, pruned_grouped_concept_ids, pruned_grouped_pam = pickle.load(open(path, "rb"))
    return concepts, ids, categories, labels, pruned_grouped_concept_ids, pruned_grouped_pam


def form_processed_fname(datastem, modelstem, ext, identifier=None):
    if identifier is None:
        return f"{datastem}_{modelstem}.{ext}"
    return f"{datastem}_{modelstem}_{identifier}.{ext}"


def form_data_dirname(datastem):
    return f"./data/{datastem}/surveys/"


def newer_groupings_to_csv(
        store_path: str, original_state: ConceptsState, state_conversions: List[Dict[int, int]],
        id_names: List[str]):
    with open(store_path, 'w') as ofile:
        line = f"{','.join(id_names)},freq,text\n"
        ofile.write(line)
        for raw_id, (text, concept_pa) in enumerate(zip(original_state.concept_strings, original_state.concept_pam.T)):
            freqs = count_datapoints_in_each_feature(concept_pa)
            curr_id = raw_id
            sols = [str(curr_id)]
            for state_conversion in state_conversions:
                curr_id = state_conversion.get(curr_id, -1)
                sols.append(str(curr_id))
            line = f"{','.join(sols[::-1])},{freqs}, {text}\n"
            ofile.write(line)


def new_groupings_to_csv(
        path: str, grouped_concept_ids: List[List[int]], all_concepts: List[str], all_concept_counts: List[int],
        pruned2groupid: Dict[int, int], simpleprune2rawid: np.ndarray):
    raw2simplepruneid = {k: i for i, k in enumerate(simpleprune2rawid)}
    simpleprune2groupid = {raw_id: group_id for group_id, raw_ids in enumerate(grouped_concept_ids) for raw_id in
                           raw_ids}
    group2prunedid = {gid: pid for pid, gid in enumerate(pruned2groupid)}

    with open(path, 'w') as ofile:
        line = "final_id,group_id,simple_prune_id,raw_id,freq,text\n"
        ofile.write(line)
        for original_id, (text, freq) in enumerate(zip(all_concepts, all_concept_counts)):
            simple_prune_id = raw2simplepruneid.get(original_id, -1)
            group_id = simpleprune2groupid.get(simple_prune_id)
            final_id = group2prunedid.get(group_id, -1)

            line = f'{final_id}, {group_id}, {simple_prune_id}, {original_id}, {freq},"{text}"\n'
            ofile.write(line)


def groupings_to_csv(
        path, grouped_concept_ids, concepts, concept_counts,
        orderby='num_mentions', pruned2groupid=None):
    if pruned2groupid is None:
        raw2prunedid = {}
    else:
        raw2prunedid = {rawid: pid
                        for pid, gid in enumerate(pruned2groupid)
                        for rawid in grouped_concept_ids[gid]}
    print(f"Writing groupings to {path}")
    # output to file
    with open(path, 'w') as ofile:
        line = "final_id,group_id,raw_id,freq,text\n"
        ofile.write(line)
        if orderby == 'num_mentions':
            grouped_concept_counts = [
                [concept_counts[id_] for id_ in ids]
                for ids in grouped_concept_ids]
            reorder = np.argsort(grouped_concept_counts)[::-1]
        elif orderby == 'num_concepts':
            num_concepts = [len(ids) for ids in grouped_concept_ids]
            reorder = np.argsort(num_concepts)[::-1]
        elif orderby == 'pruned_id':
            G = len(grouped_concept_ids)
            reorder = list(pruned2groupid) \
                      + [gid for gid in range(G) if not gid in pruned2groupid]
        else:
            raise ValueError(f"Unrecognised orderby variable: {orderby}")
        for group_id in reorder:
            for raw_id in grouped_concept_ids[group_id]:
                final_id = raw2prunedid.get(raw_id, -1)
                freq = concept_counts[raw_id]
                text = concepts[raw_id]
                line = f'{final_id},{group_id},{raw_id},{freq},"{text}"\n'
                ofile.write(line)


def get_file_info(datadir, csvfname, rows_to_remove=None):
    results = []
    with open(csvfname, 'w') as ofile:
        ofile.write("filepath,videoid\n")
        for datapoint in get_datapoint_iterator(datadir, include_fname=True, verbose=False,
                                                rows_to_remove=rows_to_remove):
            filepath = datapoint[0]
            videoid = datapoint[1]
            label = datapoint[2]
            text = datapoint[3]
            ofile.write(f"{filepath},{videoid},{label},{text}\n")
            results.append((filepath, videoid, label, text))
    return results


def load_file_to_video_csv(csvfname):
    raise NotImplemented
