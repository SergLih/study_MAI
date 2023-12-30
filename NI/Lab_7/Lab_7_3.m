%clear;
%clc;

% 3.1 Создаем обучающие множества
phi = 0.01: 0.025 : 3.14;
r = 5*cot(phi);
x = [r .* cos(phi); r .* sin(phi); phi];
xseq = con2seq(x);

plot3(x(1, :), x(2, :), x(3, :), '-r', 'LineWidth', 2);

% 3.2 Создаем сеть
net = feedforwardnet([10 2 10], 'trainlm');
net = configure(net, xseq, xseq);

% 3.3 Инициализируем весовые коэффициенты
net = init(net);
view(net);

% 3.4 задаем параметры обучения
net.trainParam.epochs = 100;
net.trainParam.goal = 1.0e-5;

% 3.5 проводим обучение
net = train(net, xseq, xseq);

% 3.7 считаем выход сети
yseq = sim(net, xseq);
y = cell2mat(yseq);

% 3.8 отображаем обучающее множество
plot3(x(1, :), x(2, :), x(3, :), '-r', y(1, :), y(2, :), y(3, :), '-b', 'LineWidth', 2);