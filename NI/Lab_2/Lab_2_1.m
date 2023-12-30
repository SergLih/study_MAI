%1.1 Построение обучающего множества
%Шаг
h = 0.025;

%Задаем координаты
X = 0:h:6;
Y = sin((X.*X) - 2*X + 3);

%Последовательность входных образцов
Pn = con2seq(Y);

%1.2
%Построение сети
delays = [1 2 3 4 5];
net = newlin([-1 1], 1, delays, 0.01);

%1.3
%Инициализация
net.inputweights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net);

%1.4 Адаптация сети
Pi = con2seq(Y(1:5));
P = Pn(6:end);
T = Pn(6:end);

for i = 1:50
    [net, Y, E] = adapt(net, P, T, Pi);
    %display(net.IW{1,1});
end;

%Ошибка обучения
err = sqrt(mse(E));
display(err);

%График ошибки
figure

E = cell2mat(Pn) - cell2mat(sim(net, Pn));
plot(X, E, 'r');

%1.5 Графики
%Эталон
figure
referenceLine = plot(X(6:end), cell2mat(T), 'r');
hold on;

%Предсказанное
approximationLine = plot(X(6:end), cell2mat(Y), 'b');
legend([referenceLine,approximationLine],'Target', 'Approximated');
hold off;