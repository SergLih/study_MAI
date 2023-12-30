%3.1 Построение обучающего множества
%Шаг
h = 0.025;
X = 0:h:6;

%Входное множество
x = sin((X.*X) - 2*X + 3);

%Эталонный выход 
Y = (1/4)*sin(X.*X -2*X);

%3.2 Расширение входного множества

%Задаем глубину погружения временного ряда
D = 4;
Q = length(x);
P = zeros(D, Q);

for i=1:D
    P(i, i:Q) = x(1:Q - i + 1);
end;

%3.3 Создаем сеть
net = newlind(P, Y);

%Отображаем полученные веса и смещение
display(net.IW{1,1});
display(net.b{1});

T = sim(net, P);
%Погрешность
E = Y - T;

figure
err = plot(X, E, 'r');
hold on;
%set(err, 'linewidth', 1);
legend(err,'Error');
hold off;

%Эталон
figure
referenceLine = plot(X, Y, 'r');
set(referenceLine, 'linewidth', 3);
hold on;

%
approximationLine = plot(X, T, '--b');
set(approximationLine, 'linewidth', 3);

legend([referenceLine,approximationLine],'Target', 'Predicted');
hold off;
