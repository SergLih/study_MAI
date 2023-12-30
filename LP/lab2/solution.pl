% Утверждения являются либо правдивыми, либо ложными.
statement(true).
statement(false).

% Один из пяти братьев разбил окно.
broke(andrey).
broke(vitya).
broke(tolya).
broke(dima).
broke(yura).

% Андрей сказал: Это или Витя, или Толя. 
hypothesis(andrey, Сauser):- Сauser=vitya; Сauser=tolya. 

% Витя сказал: Это сделал не я (т.е. не Витя) и не Юра. 
hypothesis(vitya, Сauser) :- \+(Сauser=vitya), \+(Сauser=yura). 

% Дима сказал: Нет, один из них сказал правду, а другой неправду.
hypothesis(dima, Сauser):- 
    hypothesis(andrey, Сauser),     \+(hypothesis(vitya, Сauser));
    \+(hypothesis(andrey, Сauser)),    hypothesis(vitya, Сauser).
	
% Юра сказал: Нет, Дима, ты не прав. 
hypothesis(yura, Сauser):- \+(hypothesis(dima,Сauser)).

statement(Name, Сauser, true)  :- hypothesis(Name, Сauser).
statement(Name, Сauser, false) :- \+(hypothesis(Name, Сauser)).

solve(Statements, Сauser):- broke(Сauser),
    statement(andrey, Сauser, A),
    statement(vitya,  Сauser, V),
    statement(dima,   Сauser, D),
    statement(yura,   Сauser, U),
    Statements=[A, V, D, U],
    check_statements(Statements).
	
check_statements(List) :- 
    number_true_statements(List, NumberTrue), 
    NumberTrue>=3.

true_or_false_statement(true, 1).    
true_or_false_statement(false, 0).

number_true_statements([], 0).                                  
number_true_statements([Head | Tail], C) :- 
	  C = HeadTrueOrFalse + CTail,
    number_true_statements(Tail, CTail),
    true_or_false_statement(Head, HeadTrueOrFalse).
