%clear;
%clc;

% 2.1 Генерируем обучающее множество
%phi = 0.01 : 0.025 : 2*pi;
phi = 0.01: 0.025 : 3.14;
%r = phi * 2;
r = 5*cot(phi);
x = [r .* cos(phi); r .* sin(phi)];
xseq = con2seq(x);

plot(x(1, :), x(2, :), '-r', 'LineWidth', 2);

% 2.2 Создаем многослойную сеть прямого распространения
net = feedforwardnet([10 2 10], 'trainlm');

net = configure(net, xseq, xseq);

% 2.3 Инициализируем весовые коэффициенты
net = init(net);
view(net);

% 2.4 Задаем параметры обучения
net.trainParam.epochs = 100;
net.trainParam.goal = 1.0e-5;

% 2.5 Обучение сети
net = train(net, xseq, xseq);

% 2.6 Рассчитываем выход сети
yseq = sim(net, xseq);
y = cell2mat(yseq);

% 2.7 Отображаем обучающее множество
plot(x(1, :), x(2, :), '-r', y(1, :), y(2, :), '-b', 'LineWidth', 2);