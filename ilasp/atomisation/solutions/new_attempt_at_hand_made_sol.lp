% The batter hit a fly ball to left field, which the fielder caught in the air -> go to
go_to_from(relcl).


% Must consider :
% The batter did not swing at the pitch and strike had to be called by the umpire
% the ball was quick and good -> the ball has been quick and good.
% The batter missed the ball making it a strike. Ideally -> The batter missed the ball. It a strike. (Ideally "It is a strike but we don't have that.)


% is_predicate_part(T) :- pos(verb, T).
% is_predicate_part(T) :- pos(aux, T).
% is_predicate_word(T) :- pos(part, T).
% 
% 
% 
% 
% 
% predicate_sep(conj).
% 
% 
% predicate() :- predicate_sep(X), dep(X, T1, T2).
% 0 { predicte_(T) : candidate_start(T) } 1.
% 
% 
% 
% % Let's say we have predicates in seperate answer sets
% 
% in_atomic_sent(T) :- in_atomic_sent(T2), dep(C, T2, T), not do_not_include(C).
% in_atomic_sent(T) :- dep(relcl, T, T2), in_atomic_sent(T2).
% 
% % The batter hit the ball which landed in shallow field.
% in_atomic_sent(T) :- dep(nsubj, T2, T), dep(C, T2, T), not dep(relcl, _, T2).
% 
% do_not_include(relcl).
% do_not_include(cc).
% do_not_include(nsubj).


% ----------------------------------

% The batter swung and missed, therefore it is a strike.
splitting_tag(ccomp).
% The batter did not swing so it was a ball.  
splitting_tag(advcl).
% The batter swung the bat but missed the ball.
splitting_tag(conj).
% The batter hit the ball in play where it was caught mid air by a defender.
splitting_tag(relcl).

0 { in_atomic_sent(T) } 1 :- candidate_start(T).


% Skip over rule.
in_atomic_sent(T) :- dep(conj, T1, T2), in_atomic_sent(T2), dep(_, T, T1), not candidate_start(T).

