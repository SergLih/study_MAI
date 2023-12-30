clear;
clc;

% 1.1 Генерируем обучающее множество
trange = 0 : 0.025 : 2 * pi;
x = ellipse(trange, 0.7, 0.7, 0, -0.1, -pi / 6);
xseq = con2seq(x);

plot(x(1, :), x(2, :), '-r', 'LineWidth', 2);

% 1.2 Создаем сеть
net = feedforwardnet(1, 'trainlm');
net.layers{1}.transferFcn = 'purelin';

net = configure(net, xseq, xseq);
net = init(net);
view(net);

% 1.4 задаем параметры обучения
net.trainParam.epochs = 10000;
net.trainParam.goal = 1.0e-5;

% 1.5 обучаем сеть
net = train(net, xseq, xseq);
display(net);

% 1.7 рассчитываем выход сети
yseq = sim(net, xseq);
y = cell2mat(yseq);

% 1.8 Отображаем обучающее множество
plot(x(1, :), x(2, :), '-r', y(1, :), y(2, :), '-b', 'LineWidth', 2);
%%
%r = 2;
phi = 0.01 : 0.025 : 2*pi;
r = phi * 2;
x = [r .* cos(phi); r .* sin(phi)];
xseq = con2seq(x);

plot(x(1, :), x(2, :), '-r', 'LineWidth', 2);

net = feedforwardnet([10 1 10], 'trainlm');

net = configure(net, xseq, xseq);
net = init(net);
view(net);
net.trainParam.epochs = 10000;
net.trainParam.goal = 1.0e-5;
net = train(net, xseq, xseq);

yseq = sim(net, xseq);
y = cell2mat(yseq);

plot(x(1, :), x(2, :), '-r', y(1, :), y(2, :), '-b', 'LineWidth', 2);
%%
%r = 2;
phi = 0.01 : 0.025 : 2 * pi;
r = phi * 2;
x = [r .* cos(phi); r .* sin(phi); phi];
xseq = con2seq(x);


plot3(x(1, :), x(2, :), x(3, :), '-r', 'LineWidth', 2);

net = feedforwardnet([10 2 10], 'trainlm');

net = configure(net, xseq, xseq);
net = init(net);
view(net);
net.trainParam.epochs = 10000;
net.trainParam.goal = 1.0e-5;
net = train(net, xseq, xseq);

yseq = sim(net, xseq);
y = cell2mat(yseq);

plot3(x(1, :), x(2, :), x(3, :), '-r', y(1, :), y(2, :), y(3, :), '-b', 'LineWidth', 2);