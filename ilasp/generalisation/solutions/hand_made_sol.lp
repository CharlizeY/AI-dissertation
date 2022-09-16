% the root should always be included in concept; this is how we define atomic sentences
in_generalised_sent(T) :- root(T).

% Include punctuation of there is a sentence. No commas exist in examples.
in_generalised_sent(T) :- dep(punct, T2, T), in_generalised_sent(T2).

% TODO: check PoS for verb
% nsubj should be included too if the verb is included 
in_generalised_sent(T) :- dep(nsubj, T2, T), in_generalised_sent(T2).

% dobj - direct object should always be included because the verb doesn't have a meaning without it.
% The fielder caught the ball
in_generalised_sent(T) :- dep(dobj, T2, T), in_generalised_sent(T2).

% det (the) should always be included if the word it is determinining is included
% The fielder caught the ball - keeping the grammar the same
in_generalised_sent(T) :- dep(det, T2, T), in_generalised_sent(T2).

% if a noun included then it compound should be too as it gives useful 
% It was a home run
in_generalised_sent(T) :- dep(compound, T2, T), in_generalised_sent(T2).

% we may or may not want a amod (adjectevial modifer) in our solution
% The fielder caught the ball in shallow field
0 { in_generalised_sent(T) } 1 :- dep(amod, T2, T), in_generalised_sent(T2).

% TODO - The ball crossed the plate outside of the strike zone -> Rules do not work when we have 2 preps -> A home run was hit out of the playing grounds
% if the preposition object is included then the related preposition should be too
% The fielder caught the ball in shallow field
in_generalised_sent(T) :- dep(pobj, T, T2), in_generalised_sent(T2).

% We may or may not want object of a preposition in our solution.
% The fielder caught the ball in shallow field.
0 { in_generalised_sent(T) } 1 :- dep(pobj, T2, T), dep(prep, T3, T2), in_generalised_sent(T3).

% Always include negation. We do not want to change the meaning of the sentence.
% The batter did not swing
in_generalised_sent(T) :- dep(neg, T2, T), in_generalised_sent(T2).

% Always include auxiliary (e.g. did) to preserve the meaning of the sentence.
% The batter did not swing
in_generalised_sent(T) :- dep(aux, T2, T), in_generalised_sent(T2).

% Always include passive auxiliary (e.g. was) to preserve the passive structure
% The pitch was called a strike by the umpire
in_generalised_sent(T) :- dep(auxpass, T2, T), in_generalised_sent(T2).

% Always include passive subject
% The pitch was called a strike by the umpire
in_generalised_sent(T) :- dep(nsubjpass, T2, T), in_generalised_sent(T2).

% Object predicate should always be included? - TODO
% The pitch was called a strike
in_generalised_sent(T) :- dep(oprd, T2, T), in_generalised_sent(T2).

% Attribute should always be included
% It was a home run
in_generalised_sent(T) :- dep(attr, T2, T), in_generalised_sent(T2).

% Adverbial modifier may or may not need to be in the sentence that contains 
% item it is modifying.
% The hitter hit the ball straight up the middle to the pitcher. -> This could probably work without straight
% The pitch was called a ball, rather than a strike.
in_generalised_sent(T) :- dep(advmod, T2, T), in_generalised_sent(T2).


% Conjunction may or may not be included. 
% The pitch was called a ball, rather than a strike.
0 { in_generalised_sent(T) } 1 :- dep(conj, T2, T), in_generalised_sent(T2).

% We may or may not wish to include prepositional object; the preposition may also be coordinating conjunction
% The pitch was called a ball, rather than a strike. - DisplaCy different than our code
0 { in_generalised_sent(T) } 1 :- dep(pobj, T2, T), dep(cc, T3, T2), in_generalised_sent(T3).


% We may or may not wish to include prepositional object; the preposition may also be an agent
% It was collected by the third baseman
0 { in_generalised_sent(T) } 1 :- dep(pobj, T2, T), dep(agent, T3, T2), in_generalised_sent(T3).



% Always add an acomp
% The pich was very low.
in_generalised_sent(T) :- dep(acomp, T2, T), in_generalised_sent(T2).









% --- try removing





% % We want to include the marker if the part of the sentence is included as a clause.
% The batter Holiday made contact with the pitch hitting it into the air -> does this need to be split to:  The batter Holiday made contact with the pitch. The batter Holiday hit it into the air
% in_generalised_sent(T) :- dep(mark, T2, T), in_generalised_sent(T2), dep(advcl, T3, T2), in_generalised_sent(T3).
% in_generalised_sent(T) :- dep(mark, T2, T), in_generalised_sent(T2), dep(ccomp, T3, T2), in_generalised_sent(T3), dep(advcl, T4, T3), in_generalised_sent(T4).
% 
% % We always want to include clausal complement (complement of the core argument). 
% % It may even work as a standalone sentence (The umpire behind the plate appealed to the first base umpire,who had a better view,to determine whether the batter completed a full swing.)
% The first base umpire declared that the batter did not take a full swing.
% scomp might be
% in_generalised_sent(T) :- dep(ccomp, T2, T).
% 
% 
% % We may or may not want relative clause in our solution. Done by including/not including relative clause modifier
% 0 { in_generalised_sent(T) } 1 :- dep(relcl, T2, T), in_generalised_sent(T2).
% 
% % We may or not want to include an adverbial clause
% 0 { in_generalised_sent(T) } 1 :- dep(advcl, T2, T), in_generalised_sent(T2). 
