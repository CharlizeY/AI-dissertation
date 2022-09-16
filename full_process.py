import pickle

from concept_processing import io
from concept_processing.codex_pipeline import CodexPipeline, extract_concepts
from concept_processing.legacy_code import capture_all_concepts_full_old

# store_location = "/Users/Cherry0904/Desktop/roko-for-charlize/full_dataset"
# store_location = "/Users/Cherry0904/Desktop/roko-for-charlize/short_dataset"
store_location = "/vol/bitbucket/yy3219/roko-for-charlize/2000_pn_dataset"
# store_location = "/Users/Cherry0904/Desktop/roko-for-charlize/short_dataset"

label_type = 'pn'
pruning_parameter = 0.9

use_old_pipeline = True


if __name__ == "__main__":
    if use_old_pipeline:
        state_before_grouping = capture_all_concepts_full_old(store_location)
    else:
        # original_state = extract_concepts(store_location)
        state_before_grouping = extract_concepts(store_location)
 
        # state_before_grouping = original_state
        # pipeline = CodexPipeline(methods=['simple_pruning'], use_old_pipeline=use_old_pipeline)
        # state_before_grouping, simple_pruning_conversion_dict = pipeline(original_state)

    pipeline = CodexPipeline(methods=['grouping', 'pruning'], use_old_pipeline=use_old_pipeline)
    # pipeline = CodexPipeline(methods=['grouping'], use_old_pipeline=use_old_pipeline)

    last_state, conversion_dict_list = pipeline(state_before_grouping)

    # csvfname = "/Users/Cherry0904/Desktop/roko-for-charlize/jupyter-notebooks/hyper-parameter-tuning/grouping_with_new_hyperparameters.csv"
    csvfname = f"/vol/bitbucket/yy3219/roko-for-charlize/jupyter-notebooks/hyper-parameter-tuning/{label_type}_{pruning_parameter}_grouping_with_new_hyperparameters.csv"
    io.newer_groupings_to_csv(csvfname, last_state, conversion_dict_list,
                              id_names = ["final_id", "group_id", "start_id"])
    # io.newer_groupings_to_csv(csvfname, state_before_grouping, conversion_dict_list,
    #                           id_names = ["final_id", "group_id", "start_id"])
                              # id_names = ["group_id", "start_id"])

    # We are using state after simple pruning for hyper-parameters as it is easier to construct matrix labelings that way
    # io.store_concept_objects("/vol/bitbucket/yy3219/roko-for-charlize/jupyter-notebooks/hyper-parameter-tuning/for_hyperparameter_tuning.pkl", state_before_grouping.concept_strings,
    #                         state_before_grouping.ids, state_before_grouping.label_categories,
    #                         state_before_grouping.label_indices, state_before_grouping.concept_pam)

    # io.store_concept_objects("/Users/Cherry0904/Desktop/roko-for-charlize/jupyter-notebooks/hyper-parameter-tuning/for_hyperparameter_tuning.pkl", state_before_grouping.concept_strings,
    #                          state_before_grouping.ids, state_before_grouping.label_categories,
    #                          state_before_grouping.label_indices, state_before_grouping.concept_pam)

    to_store = last_state.to_dict()
    # to_store = state_before_grouping.to_dict()
    pickle_fname = f"{label_type}_{pruning_parameter}_final_dict_{'old' if use_old_pipeline else 'new'}_codex.pkl"
    pickle.dump(to_store, open(pickle_fname, "wb"))
    stored_dict = pickle.load(open(pickle_fname, "rb"))
    print(stored_dict)
    print("Done")
