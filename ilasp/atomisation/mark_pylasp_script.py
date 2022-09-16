#ilasp_script
ilasp.cdilp.initialise()
solve_result = ilasp.cdilp.solve()

ilasp.stats.print_new_iteration()
debug_print('Searching for counterexample...')

c_egs = None
if solve_result is not None:
  c_egs = ilasp.find_all_counterexamples(solve_result)

conflict_analysis_strategy = { 
  'positive-strategy': 'single-ufs',
  'negative-strategy': 'single-as',
  'brave-strategy':    'single-ufs',
  'cautious-strategy': 'single-as-pair'
}

while c_egs and solve_result is not None:
  c_eg_ids = list(map(lambda x: x['id'], c_egs))
  while float(len(c_eg_ids))/len(c_egs) > 0.95:
    ce = ilasp.get_example(c_eg_ids[0])
    debug_print('Found', ce['type'], 'counterexample:', ce['id'], '(a total of', len(c_eg_ids), 'counterexamples found)')

    constraint = ilasp.cdilp.analyse_conflict(solve_result['hypothesis'], ce['id'], conflict_analysis_strategy)

    debug_print('Computed constraint. Now propagating to other examples...')
    prop_egs = [ce['id']]

    c_eg_ids = list(set(c_eg_ids) - set(prop_egs))
    ilasp.cdilp.add_coverage_constraint(constraint, prop_egs)
    debug_print('Constraint propagated to:', prop_egs)

  solve_result = ilasp.cdilp.solve()

  if solve_result is not None:
    debug_print('Found hypothesis:', solve_result['hypothesis'], solve_result['expected_score'])
    debug_print(ilasp.hypothesis_to_string(solve_result['hypothesis']))
    ilasp.stats.print_new_iteration()
    debug_print('Searching for counterexample...')

    c_egs = ilasp.find_all_counterexamples(solve_result)


if solve_result:
  debug_print('\n\nFinal Hypothesis:\n\n')
  print(ilasp.hypothesis_to_string(solve_result['hypothesis']))
else:
  print('UNSATISFIABLE')
  print(ilasp.cdilp.get_current_asp_program())

ilasp.stats.print_timings()

#end.


