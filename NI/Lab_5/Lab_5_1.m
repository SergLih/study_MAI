%Использование сети Элмана для распознавания динамических образов
%1.1
%Основной сигнал
k1 = 0:0.025:1;
p1 = sin(4*pi*k1);

%Целевой выход основного сигнала
t1 = -ones(size(p1));

%Сигнал подлежащий распознаванию
k2 = 1.13:0.025:3.6;
p2 = sin(sin(k2).*(k2.*k2) - k2);

%Целевой выход
t2 = ones(size(p2));

%Длительность основного сигнала
R = {7; 0; 7}; 

%Входное множество
P = [repmat(p1, 1, R{1}), p2, repmat(p1, 1, R{2}), p2, repmat(p1, 1, R{3}), p2];
T = [repmat(t1, 1, R{1}), t2, repmat(t1, 1, R{2}), t2, repmat(t1, 1, R{3}), t2];
P = con2seq(P);
T = con2seq(T);

%Создание сети
net = layrecnet(1 : 2, 8, 'trainoss');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net = configure(net, P, T);

%1.3
[p, Xi, Ai, t] = preparets(net, P, T);

%1.4
net.trainParam.epochs = 100;
net.trainParam.goal = 1.0e-5;

%1.5 Обучение
net = train(net, p, t, Xi, Ai);
Y = sim(net, p, Xi, Ai);

%1.6
view(net);

%1.7
figure;
hold on;

rLine = plot(cell2mat(Y), 'r');
pLine = plot(cell2mat(t), 'b');
legend([rLine,pLine],'Target', 'Predicted');

%1.8
tc = zeros(0, length(Y));
for i=1:length(Y)
    if Y{i} >= 0
        tc(i) = 1;
    else 
        tc(i) = -1;
    end
end

fprintf('Train set size: %d\n',length(T)-3);
T = [repmat(t1, 1, R{1}), t2, repmat(t1, 1, R{2}), t2, repmat(t1, 1, R{3}), t2];
fprintf('Successeful resulst: %d\n',nnz(tc == T(3 : end)));

%1.9
R = {7;0;7};
P = [repmat(p1, 1, R{1}), p2, repmat(p1, 1, R{2}), p2, repmat(p1, 1, R{3}), p2];
T = [repmat(t1, 1, R{1}), t2, repmat(t1, 1, R{2}), t2, repmat(t1, 1, R{3}), t2];
P = con2seq(P);
T = con2seq(T);
%Создание сети
net = layrecnet(1 : 2, 8, 'trainoss');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net = configure(net, P, T);

[p, Xi, Ai, t] = preparets(net, P, T);

net.trainParam.epochs = 100;
net.trainParam.goal = 1.0e-5;

%1.10
net = train(net, p, t, Xi, Ai);
Y = sim(net, p, Xi, Ai);

figure;
hold on;


rLine = plot(cell2mat(Y), 'r');
pLine = plot(cell2mat(t), 'b');
legend([rLine,pLine],'Target', 'Predicted');

%1.11
tc = zeros(0, length(Y));
for i=1:length(Y)
    if Y{i} >= 0
        tc(i) = 1;
    else 
        tc(i) = -1;
    end
end


fprintf('Train set size: %d\n',length(T)-3);
T = [repmat(t1, 1, R{1}), t2, repmat(t1, 1, R{2}), t2, repmat(t1, 1, R{3}), t2];
fprintf('Successeful resulst: %d\n',nnz(tc == T(3 : end)));