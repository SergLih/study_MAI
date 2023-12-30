:- consult('database.pl').	
	
person(Id, Name, _, _) :-
	person(Id, Name).
	
marriage_id(WifeId, HusbandId) :-
	mother(WifeId, Cid1), 
	father(HusbandId, Cid1),  
	Cid1 \= 0.

marriage(WifeName, HusbandName) :-
	person(WifeId, WifeName),
	person(HusbandId, HusbandName), 
	marriage_id(WifeId, HusbandId).
	
brother(Person1name, Person2name) :-
	person(Id1, Person1name),
	person(Id2, Person2name), 
	brother_id(Id1, Id2).

sibling_id(Id1, Id2) :-
	Id1 \= Id2,
	((mother(ParId1, Id1), mother(ParId1, Id2)) ; 
	(father(ParId2, Id1), father(ParId2, Id2))).
	
brother_id(Id1, Id2) :-
	father(Id1, _), 
	father(Id2, _), 
	sibling_id(Id1, Id2).
	
	
%%%%%%% Предикат поиска деверя %%%%%%%
dever_id(WifeId, DeverId) :- 
	marriage_id(WifeId, HusbandId),
	brother_id(HusbandId, DeverId).

find_devers(WifeName, ListRes) :-
	bagof((WifeName, ListRes2), (person(WifeId, WifeName), 
		setof(
			DeverName,
			DeverId^(
			person(DeverId, DeverName),
			dever_id(WifeId, DeverId)), 
			ListRes2)
	), ListRes).

	
	
/* Степень родства */

male(Person) :-
	father(Person, _).

female(Person) :-
	mother(Person, _).

check_relation_id(Person1, Person2, mother) :-
	mother(Person1, Person2).

check_relation_id(Person1, Person2, father) :-
	father(Person1, Person2).	

check_relation_id(Person1, Person2, Res) :-
	male(Person1) ->
		(male(Person2) -> 
			check_relation_id_male_male(  Person1, Person2, Res) ;
			check_relation_id_male_female(Person1, Person2, Res)
		) ;
		(male(Person2) -> 
			check_relation_id_female_male(Person1, Person2, Res) ;
			check_relation_id_female_female(Person1, Person2, Res)
		).

	
check_relation_id_male_male(Child, Parent, son) :-
    father(Parent, Child).
	
check_relation_id_male_male(Brother, Person, brother):-
	sibling_id(Person, Brother).
	
check_relation_id_male_female(Brother, Person, brother):-
	sibling_id(Person, Brother).
	
check_relation_id_male_female(Husband, Wife, husband) :-
	marriage_id(Wife, Husband).
	
check_relation_id_male_female(Child, Parent, son) :-
    mother(Parent, Child).
	
check_relation_id_female_male(Child, Parent, daughter) :-
    father(Parent, Child).
	
check_relation_id_female_male(Sister, Person, sister):-
	sibling_id(Person, Sister).

check_relation_id_female_male(Wife, Husband, wife) :-
	marriage_id(Wife, Husband).
	
check_relation_id_female_female(Child, Parent, daughter) :-
    mother(Parent, Child).	

check_relation_id_female_female(Sister, Person, sister):-
	sibling_id(Person, Sister).
	
	
check_relation(X):-
    member(X, [father, mother, sister, brother, son, daughter, husband, wife]).
	
	
	
/**********************************************************/

/*          Степень родства          */

% 1. Список людей, через которых проходит родственная связь между 2 людьми
relation_persons_list(Xname, Yname, ResNames) :-
	person(X, Xname),
	person(Y, Yname),
	relation_persons_list_id(X, Y, ResIds),
	transform_person_ids_to_names(ResIds, ResNames).

% 2. Список родственных отношений, через которые связаны два человека	
relation_list(Xname, Yname, Res) :-
	person(X, Xname),
	person(Y, Yname),
	relation_list_id(X, Y, Res).

relation_persons_list_id(X, Y, Res):- % цепочка людей, через которых связаны 2 человека
    search_bfs(X, Y, Res).
		
relation_list_id(X, Y, Res):-
    search_bfs(X, Y, Res1), !,
    transform_persons_to_relations(Res1, Res).

% вместо цепочки родственников в цепочку родства
transform_persons_to_relations([_],[]):-!.
transform_persons_to_relations([First, Second | Tail], ResList):-
    check_relation_id(First, Second, Relation),
    ResList = [Relation | Tmp],
    transform_persons_to_relations([Second | Tail], Tmp), !.
	
transform_person_ids_to_names([], []).
transform_person_ids_to_names([IdHead | Tail], Res) :-
	person(IdHead, Name),
	transform_person_ids_to_names(Tail, TransformedTail),
	Res = [Name | TransformedTail].

prolong([X | T], [Y, X | T]):-
    move(X, Y),
    \+ member(Y, [X | T]).

move(X, Y):-
    check_relation_id(X, Y, _).

	
search_bfs(StartState, EndState, ResPath):- 
	bfs([[StartState]], EndState, Path),
	reverse(Path, ResPath).

bfs([[EndState | PathTail] | _ ], EndState, [EndState|PathTail]). % ход к цели 
bfs([TempPath | OtherPaths], EndState, Path) :-  
    findall(P, prolong(TempPath, P), Paths),  
    append(OtherPaths, Paths, NewPaths), 
    bfs(NewPaths, EndState, Path), !. % достаточно найти первый (кратчайший путь)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Поиск и подсчет родственников определенного вида.
	
find_relatives(Xname, Rel, ListRes) :-
	person(Xid, Xname), 
	(setof(
		Yname,
		Yid^(
		person(Yid, Yname),
		check_relation_id(Yid, Xid, Rel)), 
		ListR) -> ListRes = ListR ; ListRes = [] ). % возвращаем пустой список если такого вида родственников нет
	
	
count_relatives(Xname, Rel, Count) :-
	person(Xid, Xname), 
	(setof(
		Yname,
		Yid^(
		person(Yid, Yname),
		check_relation_id(Yid, Xid, Rel)), 
		ListRes) ->
	length(ListRes, Count) ; Count = 0).  % возвращаем 0 если такого вида родственников нет

	
% Языковой интерфейс. Обработка естественногго языка.

qword(who) --> ["who"].
qword(how) --> ["how"].

determ(many) --> ["many"].

verb(be,    sg,  3) --> ["is"].
verb(be,    pl,  3) --> ["are"].
verb(do,    sg,  3) --> ["does"].
verb(do,    pl,  3) --> ["do"].
verb(have,  sg,  3) --> ["has"].
verb(have,  pl,  3) --> ["have"].

article(the) --> ["the"].

pronoun(of) --> ["of"].

n(X) --> [X]. % имена могут быть любыми

% sg = singular (единственное число), pl = plural (множественное)
rel(brother, sg)  --> ["brother"].
rel(sister,  sg)  --> ["sister"].
rel(mother,  sg)  --> ["mother"].
rel(father,  sg)  --> ["father"].
rel(son,     sg)  --> ["son"].
rel(daughter,sg)  --> ["daughter"].
rel(husband, sg)  --> ["husband"].
rel(wife,    sg)  --> ["wife"].


rel(brother, pl)  --> ["brothers"].
rel(sister,  pl)  --> ["sisters"].
rel(mother,  pl)  --> ["mothers"].
rel(father,  pl)  --> ["fathers"].
rel(son,     pl)  --> ["sons"].
rel(daughter,pl)  --> ["daughters"].
rel(husband, pl)  --> ["husbands"].
rel(wife,    pl)  --> ["wifes"].


% Who is the brother of X?
% Who are the brothers of X?
qWho(ResList) -->
	qword(Q), verb(V, Number1, Person), 
	article(A), rel(Rel, Number2), pronoun(P), 
	n(Zname), n(Zsurname), 
	{
		(Q==who, V==be, A==the, P==of, (Number1==Number2)), 
		(Number1 == pl ->
			(string_concat(Zname, " ", Tmp),
			 string_concat(Tmp, Zsurname, Zfullname),
			 find_relatives(Zfullname, Rel, ResList)
			 ) ;
			 (
			 string_concat(Zname, " ", Tmp),
			 string_concat(Tmp, Zsurname, Zfullname),
			 find_relatives(Zfullname, Rel, ResList)
			 )
		 )
	}.

% how many brothers does X have?	
qHowMany(Count) -->
	qword(Q), determ(M), rel(Rel, Number1), verb(V2, Number2, Person2),  
	n(Zname), n(Zsurname), verb(V3, Number3, Person3), 
	{(Q==how, M==many, V2==do, V3==have, Number1==pl, Number2==sg, Person2==3,
	  Number3==pl, Person3==3), 
	(string_concat(Zname, " ", Tmp),
	 string_concat(Tmp, Zsurname, Zfullname),
	 count_relatives(Zfullname, Rel, Count) 
	 )}.	 

q(ResList) --> qWho(ResList).
q(ResList) --> qHowMany(ResList).


ask(QueryText, Res) :-
	split_string(QueryText, " ?", " ?", Words),
	phrase(q(Res), Words).
