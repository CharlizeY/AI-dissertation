% The batter swung and missed, therefore it is a strike.
splitting_tag(ccomp).
% The batter did not swing so it was a ball.  
splitting_tag(advcl).
% The batter swung the bat but missed the ball.
splitting_tag(conj).
% The batter hit the ball in play where it was caught mid air by a defender.
splitting_tag(relcl).


% Include all incoming relationships except candiate_starts.
% This allows us to reach the predicate of a current atomic sentence.
% Atulve's ball was fast and good.
in_atomic_sent(T) :- dep(_, T, T2), in_atomic_sent(T2), not candidate_start(T).


% Incoming relatinship to the first conjunct should also be included for the second one.
% This holds for conj only. The clauses tend to be self-sufficient.
% Atulve's ball was good and quick. -> Atulve's ball was quick. Atulve ball was good.
in_atomic_sent(T) :- dep(_, T, T2), dep(conj, T2, T3), in_atomic_sent(T3), not candidate_start(T).

% Include all chilren tags apart from those that are blacklisted (we do not want and, therefore...)
% Additionally, we do not want to include a candidate_start token. 
%   Atulve's ball is fast and good. Predicate "is" should not include "fast".

% Advmod handled later
do_not_include(advmod).
% The batter hit the ball and it landed far away.
do_not_include(cc).
% Not including punctuation at the moment, but preserving it later may be needed.
do_not_include(punct).
% The umpire ruled that the batter did not swing.
do_not_include(mark).
do_not_include(C) :- splitting_tag(C).

in_atomic_sent(T) :- dep(C, T2, T), in_atomic_sent(T2), not do_not_include(C), not candidate_start(T).

% Advmod rule
% We want it to distinguish between qualifiers and adverbials in a narrow sense (modifiers of predicates). Helps, very slightly (+ 0.0025 jaccard) so commented out for now
% The batter swung and missed, therefore it is a strike. | A ball is too high. 
% Advmod does not: https://universaldependencies.org/u/dep/advmod.html
% is_predicate(T) :- root(T).
% is_predicate(T) :- is_predicate(T2), dep(C, T2, T), splitting_tag(C).
% in_atomic_sent(T) :- dep(advmod, T2, T), in_atomic_sent(T2), not candidate_start(T), not is_predicate(T2).


% Include the subject of the first conjunct as a part of the second sentence if it does not contain
% its own subject.
% The batter swung but missed the ball -> The batter swung. The batter missed the ball.
in_atomic_sent(T) :- dep(nsubj, T1, T), dep(C, T1, T2), splitting_tag(C), in_atomic_sent(T2), not adjacent_subj.

