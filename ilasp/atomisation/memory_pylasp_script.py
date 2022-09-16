#ilasp_script
import os

def analyse_cdpi_conflict(ce):

  constraint = {'disjunction': list(propagated_schemas[ce['id']]) }
  new_schemas = []
  debug_print('Analysing', ce['id'], '(initially', len(constraint['disjunction']), 'schemas).')

  schema = ilasp.cdilp.generate_sufficient_constraint(ce['id'], {'negation': constraint})

  while schema is not None:
    constraint['disjunction'].append(schema)
    new_schemas.append(schema)
    schema = ilasp.cdilp.generate_sufficient_constraint(ce['id'], {'negation': constraint})

  props = ilasp.cdilp.propagate_constraints_pl(new_schemas, all_examples, {'select-examples': ['positive', 'negative'], 'strategy': 'constraint-implies-cdpi'})
  for (disjunct,egs) in props:
    neg_ids = []
    for eg_id in egs:
      propagated_schemas[eg_id].append(disjunct)
      if ilasp.get_example(eg_id)['type'] == 'negative':
        neg_ids.append(eg_id)

    if neg_ids:
      ilasp.cdilp.add_coverage_constraint({'negation': disjunct}, neg_ids)


  debug_print('Translated', ce['id'], 'into', len(constraint['disjunction']), 'schemas.')

  if ce['type'] == 'positive':
    return constraint
  else:
    return {'negation': constraint}


def analyse_cdoe_conflict(ce):

  constraint = {'disjunction': []}

  schema = ilasp.cdilp.generate_sufficient_constraint(ce['id'], {'negation': constraint})

  while schema is not None:
    constraint['disjunction'].append(schema)
    schema = ilasp.cdilp.generate_sufficient_constraint(ce['id'], {'negation': constraint})

  if ce['type'] == 'brave-order':
    return constraint
  else:
    return {'negation': constraint}


ilasp.cdilp.initialise()
all_examples = ilasp.all_examples()
propagated_schemas = {}
for eg in all_examples:
  propagated_schemas[eg] = []
solve_result = ilasp.cdilp.solve()

ilasp.stats.print_new_iteration()
debug_print('Searching for counterexample...')

c_egs = None
if solve_result is not None:
  c_egs = ilasp.find_all_counterexamples(solve_result)

conflict_analysis_strategy = {
  'positive-strategy': 'all-ufs',
  'negative-strategy': 'single-as',
  'brave-strategy':    'all-ufs',
  'cautious-strategy': 'single-as-pair'
}

max_memory_percent = 80 # 80% of available memory
best_solve_result = solve_result
best_score = sum(list(map(lambda x: x['penalty'], c_egs)))
total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])

while c_egs and solve_result is not None and (100 * used_memory / total_memory) < max_memory_percent:
  ce = ilasp.get_example(c_egs[0]['id'])
  debug_print('Found', ce['type'], 'counterexample:', ce['id'], '(a total of', len(c_egs), 'counterexamples found)')

  constraint = {}
  if ce['type'] == 'negative':
    constraint = ilasp.cdilp.analyse_conflict(solve_result['hypothesis'], ce['id'], conflict_analysis_strategy)
  else:
    constraint = analyse_cdpi_conflict(ce) if (ce['type'] == 'positive') else analyse_cdoe_conflict(ce)

  # An example with recorded penalty of 0 is in reality an example with an
  # infinite penalty, meaning that it must be covered. Constraint propagation is,
  # therefore, unnecessary.
  if not ce['penalty'] == 0:
    c_eg_ids = list(map(lambda x: x['id'], c_egs))
    debug_print('Computed constraint. Now propagating to other examples...')
    prop_egs = []
    if ce['type'] == 'positive':
      prop_egs = ilasp.cdilp.propagate_constraint(constraint, c_eg_ids, {'select-examples': ['positive'], 'strategy': 'cdpi-implies-constraint'})
    elif ce['type'] == 'brave-order':
      prop_egs = ilasp.cdilp.propagate_constraint(constraint, c_eg_ids, {'select-examples': ['brave-order'],    'strategy': 'cdoe-implies-constraint'})
    elif ce['type'] == 'negative':
      prop_egs = ilasp.cdilp.propagate_constraint(constraint, c_eg_ids, {'select-examples': ['negative'], 'strategy': 'neg-constraint-implies-cdpi'})
    else:
      prop_egs = [ce['id']]

    ilasp.cdilp.add_coverage_constraint(constraint, prop_egs)
    debug_print('Constraint propagated to:', prop_egs)

  else:
    ilasp.cdilp.add_coverage_constraint(constraint, [ce['id']])

  solve_result = ilasp.cdilp.solve()

  if solve_result is not None:
    debug_print('Found hypothesis:', solve_result['hypothesis'], solve_result['expected_score'])
    debug_print(ilasp.hypothesis_to_string(solve_result['hypothesis']))
    print("", flush=True)
    ilasp.stats.print_new_iteration()
    debug_print('Searching for counterexample...')

    c_egs = ilasp.find_all_counterexamples(solve_result)
    score = solve_result['expected_score'] + sum(list(map(lambda x: x['penalty'], c_egs)))
    if best_score == -1 or best_score > score:
      best_score = score
      best_solve_result = solve_result

  total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])

if solve_result:
  debug_print('\n\nFinal Hypothesis:\n\n')
  print(ilasp.hypothesis_to_string(best_solve_result['hypothesis']))
else:
  print('UNSATISFIABLE')

ilasp.stats.print_timings()

#end.

