% Реализация стандартных предикатов обработки списков.

% Вычисление длины списка.
% (список, длина)
my_length([], 0).
my_length([_ | Y], N):-my_length(Y, N1), N is N1 + 1 .

% Определение принадлежности элемента списка.
% (элемент, список)
my_member(X, [X | _]).
my_member(X, [_ | Y]):-my_member(X, Y).

% Конкатенация списков.
% (список1, список2, результат конкатенации этих двух списков)
my_append([], X, X).
my_append([Lhead|Ltail],X,[Lhead|ResTail]) :- my_append(Ltail,X,ResTail).

% Удаление элемента из списка.
% (указанный элемент, список, список без указанного элемента)
my_remove(_, [], []).
my_remove(X, [X], []).
my_remove(X, [X | T], T).
my_remove(X, [Y | T], [Y | T1]) :- my_remove(X, T, T1).

% Перестановки элементов в списке.
% (список, результат перестановки элементов списка)
my_permute([], []).
my_permute(L, [X | R]):-my_permute(T, R), my_remove(X, L, T).

% Подсписки списка.
% (подсписок, список)
my_sublist(S, L):-my_append(_, L2, L), my_append(S, _, L2).

% ОБРАБОТКА СПИСКОВ.

% Вставка элемента в список на указанную позицию (без стандартных предикатов).
% (список, элемент, указанная позиция, результат после вставки элемента в список)

my_insert(L, A, 1, [A | L]):- !.
my_insert([], A, _, R) :- R = [A], !.
my_insert(L, _, N, R) :- N =< 0, R = L, !.
my_insert([L1 | L0], A, N, R) :- Nn is N-1, my_insert(L0, A, Nn, R0), R = [L1 | R0].

% Вставка элемента в список на указанную позицию (с использованием стандартных предикатов).
% (список, элемент, указанная позиция, результат после вставки элемента в список)

ins(L, _, Pos, Res) :- length(L, Len), Pos > Len, Res = L, !. % если позиция больше длины списка или
ins(L, _, Pos, Res) :- Pos =< 0, Res = L, !.                  % меньше 0, то список не меняется
ins(L, Elem, Pos, Res) :-
  PrefixLength is Pos - 1,
    length(Prefix, PrefixLength),
    append(Prefix, Suffix, L),
    append(Prefix, [Elem], Temp),
    
    append(Temp, Suffix, Res).
    
% ОБРАБОТКА ЧИСЛОВЫХ СПИСКОВ.

% Вычисление позиции первого отрицательного элемента в списке.
% (список, определение позиции первого отриц-го эл-та в списке)
find_negative_elem([], X) :- X = 0 . 
find_negative_elem([N|_], X) :- N < 0, X = 1, !.
find_negative_elem([_ | Tail], X) :- find_negative_elem(Tail, Y), Y > 0, X is Y + 1 , !.
find_negative_elem([_ | Tail], X) :- find_negative_elem(Tail, Y), Y == 0, X is 0 .  % если не найдено, возвращаем 0


% Пример совместного использования предикатов.
% Вставка нового элемента на указанную позицию и после этого - вычислить позицию первого отриц-го элемента полученного списка.
% (входной список, новый элемент, позиция нового элемента, позиция первого отриц.элемента полученного списка)
find_negative_elem_of_new_list(L1, Y, Pos, Res):- my_insert(L1, Y, Pos, L), find_negative_elem(L, Res).