import timeit

from numpy import savetxt
from concept_processing.codex_pipeline import CodexPipeline

from concept_processing.legacy_code import capture_all_concepts_full_old
from concept_processing.pam import calc_cummulative_mi

concept_maxK = 500
datapath2 = "/vol/bitbucket/yy3219/roko-for-charlize/artemis_dataset/subset_mi_test_1500"
datapath3 = "/vol/bitbucket/yy3219/roko-for-charlize/artemis_dataset/subset_mi_test_2000"
datapath4 = "/vol/bitbucket/yy3219/roko-for-charlize/artemis_dataset/subset_mi_test_2500"
datapath5 = "/vol/bitbucket/yy3219/roko-for-charlize/artemis_dataset/subset_mi_test_3000"

# stop1 = timeit.default_timer()

# initial_state_old = capture_all_concepts_full_old(datapath2)
# pipeline = CodexPipeline(methods=['grouping', 'pruning'], use_old_pipeline=True)
# after_grouping_old, _ = pipeline(initial_state_old)
# old_pruning_cummulative_mi_1500, _ = calc_cummulative_mi(after_grouping_old.concept_pam,
#                                                      label_ids=after_grouping_old.label_indices, threshold=0,
#                                                      maxK=concept_maxK)

# stop2 = timeit.default_timer()

# initial_state_old = capture_all_concepts_full_old(datapath3)
# pipeline = CodexPipeline(methods=['grouping', 'pruning'], use_old_pipeline=True)
# after_grouping_old, _ = pipeline(initial_state_old)
# old_pruning_cummulative_mi_2000, _ = calc_cummulative_mi(after_grouping_old.concept_pam,
#                                                      label_ids=after_grouping_old.label_indices, threshold=0,
#                                                      maxK=concept_maxK)

# stop3 = timeit.default_timer()

# initial_state_old = capture_all_concepts_full_old(datapath4)
# pipeline = CodexPipeline(methods=['grouping', 'pruning'], use_old_pipeline=True)
# after_grouping_old, _ = pipeline(initial_state_old)
# old_pruning_cummulative_mi_2500, _ = calc_cummulative_mi(after_grouping_old.concept_pam,
#                                                      label_ids=after_grouping_old.label_indices, threshold=0,
#                                                      maxK=concept_maxK)

stop4 = timeit.default_timer()

initial_state_old = capture_all_concepts_full_old(datapath5)
pipeline = CodexPipeline(methods=['grouping', 'pruning'], use_old_pipeline=True)
after_grouping_old, _ = pipeline(initial_state_old)
old_pruning_cummulative_mi_3000, _ = calc_cummulative_mi(after_grouping_old.concept_pam,
                                                     label_ids=after_grouping_old.label_indices, threshold=0,
                                                     maxK=concept_maxK)

stop5 = timeit.default_timer()


savetxt('/vol/bitbucket/yy3219/roko-for-charlize/cum_mi_3000.csv', old_pruning_cummulative_mi_3000, delimiter=',')

# print('Time for 1500: ', stop2 - stop1) 
# print('Time for 2000: ', stop3 - stop2) 
# print('Time for 2500: ', stop4 - stop3) 
print('Time for 3000: ', stop5 - stop4) 