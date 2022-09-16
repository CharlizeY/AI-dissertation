% try removing any reference to a sentence, we probably don't need anything.

% BACKGROUND

% types
token(T) :- start(T).
token(T) :- succ(_, T).
token(T) :- succ(T, _).
token(T) :- dep(_, T, _).
token(T) :- dep(_, _, T).
label(L) :- dep(L, _, _).
entity(X) :- token(X).
entity(X) :- label(X).
1 { token(X); label(X) } 1 :- entity(X).

% sequences
% only one token can start a sentence
:- start(T0), start(T1), T0 != T1.
% a token appears after another if there is a string of successors.

% dependencies
% a dependent token can only be a dependency of one type
:- dep(L0, _, T), dep(L1, _, T), L0 != L1.
% a typed dependency only has one
:- dep(L, T0, T), dep(L, T1, T), T0 != T1.
% a token cannot be a dependency of itself
:- dep(_, T, T).

word(W) :- token(_, W).


chain(T1, T2) :- dep(_, T1, T2).
chain(T1, T3) :- dep(_, T1, T2), chain(T2, T3).

% T2 is reachable from T1 by a path using in_generalised_sent nodes
in_generalised_sent_chain(T1, T2) :- in_generalised_sent(T1), in_generalised_sent(T2), dep(_, T1, T2).
in_generalised_sent_chain(T1, T3) :- in_generalised_sent_chain(T1, T2), in_generalised_sent_chain(T2, T3).


:- in_generalised_sent(T2), root(T1), T1 != T2, not in_generalised_sent_chain(T1, T2).


% CUSTOM stuff


% the root should always be included in concept; this is how we define atomic sentences
in_generalised_sent(T) :- root(T).

% there must be a token in any concept sentence
:- {in_generalised_sent(T) : token(T)} = 0.

% TODO: check PoS for verb
% nsubj should be included too if the verb is included 
in_generalised_sent(T) :- dep(nsubj, T2, T).

% dobj - direct object should always be included because the verb doesn't have a meaning without it.
% The fielder caught the ball
in_generalised_sent(T) :- dep(dobj, T2, T).

% det (the) should always be included if the word it is determinining is included
% The fielder caught the ball - keeping the grammar the same
in_generalised_sent(T) :- dep(det, T2, T).

% if a noun included then it compound should be too as it gives useful 
% It was a home run
in_generalised_sent(T) :- dep(compound, T2, T).

% we may or may not want a amod (adjectevial modifer) in our solution
% The fielder caught the ball in shallow field
0 { in_generalised_sent(T) } 1 :- dep(amod, T2, T).


in_generalised_sent(T) :- dep(pobj, T, T2).


% We may or may not want object of a preposition in our solution.
% The fielder caught the ball in shallow field.
0 { in_generalised_sent(T) } 1 :- dep(pobj, T2, T).

% The related preposition should be included if the pobj is too.
% The fielder caught the ball in shallow field. Written as always include a prep, eliminate answer sets that do not always contain a path.
0 { in_generalised_sent(T) } 1 :- dep(prep, T2, T).



% TODO - The ball crossed the plate outside of the strike zone -> Rules do not work when we have 2 preps -> A home run was hit out of the playing grounds
% if the preposition object is included then the related preposition should be too
% The fielder caught the ball in shallow field
in_generalised_sent(T) :- dep(pobj, T, T2).

% We may or may not want object of a preposition in our solution.
% The fielder caught the ball in shallow field.
0 { in_generalised_sent(T) } 1 :- dep(pobj, T2, T), dep(prep, T3, T2).

% Always include negation. We do not want to change the meaning of the sentence.
% The batter did not swing
in_generalised_sent(T) :- dep(neg, T2, T).

% Always include auxiliary (e.g. did) to preserve the meaning of the sentence.
% The batter did not swing
in_generalised_sent(T) :- dep(aux, T2, T).

% Always include passive auxiliary (e.g. was) to preserve the passive structure
% The pitch was called a strike by the umpire
in_generalised_sent(T) :- dep(auxpass, T2, T).

% Always include passive subject
% The pitch was called a strike by the umpire
in_generalised_sent(T) :- dep(nsubjpass, T2, T).

% Object predicate should always be included? - TODO
% The pitch was called a strike
in_generalised_sent(T) :- dep(oprd, T2, T).

% Attribute should always be included
% It was a home run
in_generalised_sent(T) :- dep(attr, T2, T).

% Adverbial modifier may or may not need to be in the sentence that contains 
% item it is modifying.
% The hitter hit the ball straight up the middle to the pitcher. -> This could probably work without straight
% The pitch was called a ball, rather than a strike.
in_generalised_sent(T) :- dep(advmod, T2, T).


% Conjunction may or may not be included. 
% The pitch was called a ball, rather than a strike.
0 { in_generalised_sent(T) } 1 :- dep(conj, T2, T).

% We may or may not wish to include prepositional object; the preposition may also be coordinating conjunction
% The pitch was called a ball, rather than a strike. - DisplaCy different than our code
0 { in_generalised_sent(T) } 1 :- dep(pobj, T2, T), dep(cc, T3, T2).


% We may or may not wish to include prepositional object; the preposition may also be an agent
% It was collected by the third baseman
0 { in_generalised_sent(T) } 1 :- dep(pobj, T2, T), dep(agent, T3, T2).


