:- include('database.pl').

my_member(A, [A|_]).
my_member(A, [_|T]) :- 
		my_member(A, T).

my_member(A, [A|_], _, _).
my_member(A, [_|T], _, _) :- 
	my_member(A, T, _, _).
 
parse_phrase([Question, Verb, Agent, '?'], [Meaning, Infinitive, animate(Agent) ]):- 
	question(Question, Meaning,  _, []),
	verb(Verb, Infinitive, _, []),
	agent(Agent, _, []).

parse_phrase([Question, Verb, Object, '?'], [Meaning, Infinitive, inanimate(Object)]):-
	question(Question, Meaning,  _, []),
	verb(Verb, Infinitive, _, []),
	object(Object, _, []).
	

grammar(Expr, Parsed):-
	grammar(Expr, Parsed,  _ , []).
	
grammar([Question, Verb, Agent, ?], [Meaning, Infinitive, (Agent, Anim)]) -->
	question(Question, Meaning),
	verb(Verb, Infinitive),
	anim(Agent, Anim),
	questionmark.

questionmark --> [?].
	
test --> [test], questionmark.
