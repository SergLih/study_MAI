% 2.1 Создаем обучающее множество
k1 = 0 : 0.025 : 1;
p1 = sin(4 * pi * k1);
t1 = -ones(size(p1));

k2 = 1.13:0.025:3.6;
g = @(k)sin(sin(k2).*(k2.*k2) - k2);
p2 = g(k2);
t2 = ones(size(p2));
R = {7; 0; 7};
P = [repmat(p1, 1, R{1}), p2, repmat(p1, 1, R{2}), p2, repmat(p1, 1, R{3}), p2];
T = [repmat(t1, 1, R{1}), t2, repmat(t1, 1, R{2}), t2, repmat(t1, 1, R{3}), t2];
Pseq = con2seq(P);
Tseq = con2seq(T);

% 2.2 создаем сеть
net = distdelaynet({0 : 4, 0 : 4}, 8, 'trainoss');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.divideFcn = '';

% 2.3 конфигурируем сеть
net = configure(net, Pseq, Tseq);
view(net);

% 2.4 формируем массивы ячеек для функции обучения
[Xs, Xi, Ai, Ts] = preparets(net, Pseq, Tseq); 

% 2.5 задаем параметры обучения
net.trainParam.epochs = 100;
net.trainParam.goal =  1.0e-5;

% 2.6 обучаем сеть
net = train(net, Xs, Ts, Xi, Ai);

% 2.8 рассчитываем выход сети
Y = sim(net, Xs, Xi, Ai);
%%
figure;
hold on;
grid on;
plot(cell2mat(Ts), '-b');
plot(cell2mat(Y), '-r');
%plot([cell2mat(Xi) cell2mat(Y)], '-r');
%figure;
%plot([cell2mat(Xi) cell2mat(Y) -cell2mat(Tseq)]);

% 2.9 преобразовываем значения по правилу
Yc = zeros(1, numel(Y));
for i = 1 : numel(Y)
    if Y{i} >= 0
        Yc(i) = 1;
    else
        Yc(i) = -1;
    end
end
accuracy = mean(Yc==cell2mat(Ts))