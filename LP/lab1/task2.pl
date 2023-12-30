% Task 2: Relational Data

% Часть 2
% Вариант представления: two.pl
% Вариант заданий: 1

% The line below imports the data
:- ['two.pl'].

% Печать строки.
print_str([]).
print_str([Char|T]):-
	put(Char),
	print_str(T).	

% 1) Получить таблицу групп и средний балл по каждой из групп.

% Печать таблицы групп и среднего балла по каждой из групп.
% (номер группы, средний балл)
print_group_avg_grade([], []).
print_group_avg_grade([GroupHead|GroupTail], [GradeHead|GradeTail]):-
	print_str("Группа: "), write(GroupHead), tab(1), print_str("Средний балл: "), write(GradeHead), nl,
	print_group_avg_grade(GroupTail, GradeTail).

% Средний балл по каждой из групп.
% (список оценок, средний балл)
avg([], 0).
avg(X, R) :- sum1(X, S), length(X, L), R is S / L.

% Сумма оценок.
% (список оценок, сумма оценок)
sum1([],0).
sum1([X|T],Sum):-
	sum1(T,Sum1),
	Sum is Sum1 + X.

% Таблица групп и средний балл по каждой из групп.
% (номер группы, средний балл)
groupGrades(GroupNumber, AvgGrade) :- 
	bagof(TheGrade, Name^Subject^grade(GroupNumber, Name, Subject, TheGrade), AllGrades), 
	avg(AllGrades, AvgGrade).

task1 :-
	findall(GroupNumber, groupGrades(GroupNumber, AvgGrade), Groups),
	findall(AvgGrade, groupGrades(GroupNumber, AvgGrade), AvgGrades),
	print_group_avg_grade(Groups, AvgGrades).


% 2) Для каждого предмета получить список студентов, не сдавших экзамен (grade=2).

% Печать предмета и студентов, не сдавших экзамен по каким-либо предметам.
% (название предмета, имя студента)
print_subj_names([], []).
print_subj_names([SubjHead|SubjTail], [NameHead|NameTail]):-
	print_str("Предмет: "), write(SubjHead), tab(3), print_str("Не сдавшие: "), write(NameHead), nl,
	print_subj_names(SubjTail, NameTail).

% Список студентов, не сдавших экзамен.
% (название предмета, имя студента)
subjectLoosers(Subject, Loosers):-
	bagof(Name, GroupNumber^grade(GroupNumber, Name, Subject, 2), Loosers).
	
task2 :- 
	findall(Subject, subjectLoosers(Subject, Loosers), Subjects),
	findall(Loosers, subjectLoosers(Subject, Loosers), LoosersBySubjects),
	print_subj_names(Subjects, LoosersBySubjects).
  

% 3) Найти количество несдавших студентов в каждой из групп.

% Печать номеров групп и кол-ва несдавших студентов
% (номер группы, число студентов)
print_group_names([], []).
print_group_names([GroupHead|GroupTail], [LC_Head|LC_Tail]):-
	print_str("Группа: "), write(GroupHead), tab(2), 
	print_str("Кол-во не сдавших: "), write(LC_Head), nl,
	print_group_names(GroupTail, LC_Tail).

% Кол-во несдавших студентов
% (номер группы, число несдавших)
groupLoosers(GroupNumber, LoosersCount):-
	bagof(Name, Subject^grade(GroupNumber, Name, Subject, 2), Loosers), length(Loosers, LoosersCount).
	
task3 :-
	findall(GroupNumber, groupLoosers(GroupNumber, LoosersCount), GroupNumbers),
	findall(LoosersCount, groupLoosers(GroupNumber, LoosersCount), LoosersBySubjects),
	print_group_names(GroupNumbers, LoosersBySubjects).
