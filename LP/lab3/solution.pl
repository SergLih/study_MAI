:- use_module(library(statistics)).

% Команда для запуска поиска в глубину
start_dfs :-
	search_dfs([boatL, [3,3], [0,0]], [_, [0,0], [3,3]]).
	
% Команда для запуска поиска в ширину
start_bfs :-
	search_bfs([boatL, [3,3], [0,0]], [_, [0,0], [3,3]]).

% Команда для запуска поиска в глубину с итеративным погружением
start_id :-
	search_id([boatL, [3,3], [0,0]], [_, [0,0], [3,3]]).

print_path([]).
print_path([A|T]) :- print_path(T), write(A), nl.

safe(M, C) :- M*M >= M*C.
boat_safe(M, C) :- M >= 0, C >= 0, 1 =< M+C, M+C =< 3, safe(M, C).
									  
step([boatL, [ML0, CL0], [MR0, CR0]], [boatR, [ML1, CL1], [MR1, CR1]]) :-
  safe(ML0, CL0),
	safe(MR0, CR0),
	member(DM, [0, 1, 2, 3]),
	member(DC, [0, 1]),
	
	boat_safe(DM, DC),
	DM =< ML0,  DC =< CL0, 

	ML1 is ML0 - DM,
	CL1 is CL0 - DC, 
	MR1 is MR0 + DM, 
	CR1 is CR0 + DC.
	
step([boatR, StL0, StR0], [boatL, StL1, StR1]) :-
	step([boatL, StR0, StL0], [boatR,  StR1,  StL1]).

	
prolong([CurState | PathTail], [NewState, CurState | PathTail]):-
    step(CurState, NewState),
    not(member(NewState, [CurState | PathTail])).

search_dfs(StartState, EndState):-
    statistics(walltime, _),
	
    dfs([StartState], EndState, ResPath),	
	                        
    statistics(walltime, [_ | [ExecutionTime]]),
    print_path(ResPath),
    write('Time '), write(ExecutionTime), write(' msec.'), nl.


search_bfs(StartState, EndState):- 
	statistics(walltime, _),
	
	bfs([[StartState]], EndState, ResPath),
	
	statistics(walltime, [_ | [ExecutionTime]]),
	print_path(ResPath),
	write('Time '), write(ExecutionTime), write(' msec.'), nl.
	
	
dfs([EndState | PathTail], EndState, [EndState | PathTail]).
dfs(CurPath, EndState, ResPath) :-
    prolong(CurPath, NewPath),
    dfs(NewPath, EndState, ResPath).


bfs([[EndState | PathTail] | _ ], EndState, [EndState|PathTail]).
bfs([TempPath | OtherPaths], EndState, Path) :-  
    findall(P, prolong(TempPath, P), Paths),  
    append(OtherPaths, Paths, NewPaths), 
    bfs(NewPaths, EndState, Path).

generate_d(1).
generate_d(N):-generate_d(M), N is M + 1.	

search_id(A,B) :-
    statistics(walltime, _),
    generate_d(DL),
    id([A],B,P,DL),
    statistics(walltime, [_ | [ExecutionTime]]),
	print_path(P),
    write('Time '), write(ExecutionTime), write(' msec.'), nl.
	
id([EndState | PathTail],EndState , [EndState | PathTail], 0).
id(TempPath, EndState , Path, N):-
    N > 0,
    prolong(TempPath, NewPaths),
    N1 is N - 1,
    id( NewPaths, EndState , Path, N1).
