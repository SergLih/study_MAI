format compact
% 1.1 Генерируем обучающее множество
trange = 0 : 0.025 : 2 * pi;
%x = ellipse(trange, 0.7, 0.7, 0, -0.1, -pi / 6);
x = cell2mat(arrayfun(@(t) rectangle(t, 0.6, 0.2, 0.1, 3.5, 3*pi/4, 0.001), trange, 'UniformOutput',false));

hold on
xline(0.0, '--b');
yline(0.0, '--b');
grid on
axis equal
plot(x(1, :), x(2, :), '-r', 'LineWidth', 2)
hold off

xseq = con2seq(x);

%1.2 Создаем сеть
net = feedforwardnet(1, 'trainlm');
net.layers{1}.transferFcn = 'purelin';

net = configure(net, xseq, xseq);
net = init(net);
view(net);

% 1.4 задаем параметры обучения
net.trainParam.epochs = 100;
net.trainParam.goal = 1.0e-5;

% 1.5 обучаем сеть
net = train(net, xseq, xseq);
display(net);

% 1.7 рассчитываем выход сети
yseq = sim(net, xseq);
y = cell2mat(yseq);

% 1.8 Отображаем обучающее множество
hold on
xline(0.0, '--b');
yline(0.0, '--b');
grid on
axis equal
plot(x(1, :), x(2, :), '-r', y(1, :), y(2, :), '-b', 'LineWidth', 2);
hold off