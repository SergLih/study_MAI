clear;
clc;

% 1.1 - 1.2 считываем данные из файла
fileID = fopen('wolf_dataset.txt','r');
formatSpec = '%f %f';
sizeData = [1 Inf];
wolf_dataset = fscanf(fileID,formatSpec,sizeData);
fclose(fileID);

% 1.3 сглаживание
x = wolf_dataset;
x = smooth(x, 12);

% 1.4 глубина погружения временного ряда
D = 5;
ntrain = 500;
nval = 100;
ntest = 50;

% 1.5 объединяем подмножества в обучающую выборку
trainInd = 1 : ntrain; % 1..500
valInd = ntrain + 1 : ntrain + nval; % 501..600
testInd = ntrain + nval + 1 : ntrain + nval + ntest; % 601..650

% 1.7 создаем сеть
%net = timedelaynet(1:D,10,'trainlm'); %1:10 delay, hid. l. size.
net = timedelaynet(1:D,8,'trainlm'); %1:10 delay, hid. l. size.
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

% 1.6 преобразуем обучающее множество
x = con2seq(x(1:ntrain+nval+ntest)'); %'

% 1.9 конфигурируем сеть под обучающее множство
net = configure(net, x, x);

% 1.10 инициализируем весовые коэфф
net = init(net);

% 1.11 задаем параметры обучения
net.trainParam.epochs = 2000;
net.trainParam.max_fail = 2000;
net.trainParam.goal = 1.0e-5;
view(net);

% 1.12 обучаем сеть
[Xs, Xi, Ai, Ts] = preparets(net, x, x); 
net = train(net, Xs, Ts, Xi, Ai);

% 1.14 рассчитываем выход сети
Y = sim(net, Xs, Xi);
%%
figure;
hold on;
grid on;
plot(cell2mat(x), '-b');
plot([cell2mat(Xi) cell2mat(Y)], '-r');

%%

figure;
hold on;
grid on;
plot([cell2mat(Xi) cell2mat(Y)] - cell2mat(x), '-r');
%%
xm = cell2mat(x);
ym = cell2mat(Y);

figure;
hold on;
grid on;
plot(xm(ntrain + nval + 1 : ntrain + nval + ntest), '-b');
plot(ym(ntrain + nval - 9 : ntrain + nval + ntest - 10), '-r');


%%
% 2.1 Создаем обучающее множество
k1 = 0 : 0.025 : 1;
p1 = sin(4 * pi * k1);
t1 = -ones(size(p1));
k2 = 0.01 : 0.025 : 2.98;
g = @(k)cos(cos(k) .* k.^2 - k);
%g = @(k)sin(-3 * k .* k + 5 * k + 10) + 0.8;
p2 = g(k2);
t2 = ones(size(p2));
%R = {2; 1; 5};
R = {1; 4; 7};
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
net.trainParam.epochs = 1000;
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
%%display(nnz(Yc == T(3 : end)))

% 3.1 строим обучающее множество
t0 = 0;
tn = 10;
dt = 0.01;
n = (tn - t0) / dt + 1;
%fun = @(k)sin(k.^2) + sin(k).^2;
fun = @(k)sin(k.^2 - 10 * k + 3);
fun2 = @(y, u)y ./ (1 + y.^2) + u.^3;
u = zeros(1, n);
u(1) = fun(0);
x = zeros(1, n);
for i = 2 : n
    t = t0 + (i - 1) * dt;
    x(i) = fun2(x(i - 1), u(i - 1));
    u(i) = fun(t);
end

figure
subplot(2,1,1)
plot(t0:dt:tn, u, '-b'),grid
ylabel('control')
subplot(2,1,2)
plot(t0:dt:tn, x, '-r'), grid
ylabel('state')
xlabel('t')

% 3.2 Глубина погружения временного ряда
D = 3;
ntrain = 700;
nval = 200;
ntest = 97;

% 3.3 объединяем подмножества в обучающую выборку
trainInd = 1 : ntrain;
valInd = ntrain + 1 : ntrain + nval;
testInd = ntrain + nval + 1 : ntrain + nval + ntest;

% 3.5 создаем NARX сеть
net = narxnet(1 : 3, 1, 8);
net.trainFcn = 'trainlm';

% 3.6 разделяем обучающее множество
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

% 3.9 задаем параметры обучения
net.trainParam.epochs = 2000;
net.trainParam.max_fail = 2000;
net.trainParam.goal = 1.0e-5;

[Xs, Xi, Ai, Ts] = preparets(net, con2seq(u), {}, con2seq(x)); 

% 3.10 проводим обучение сети
net = train(net, Xs, Ts, Xi, Ai);
view(net);

% 3.12 рассчитываем выход сети
Y = sim(net, Xs, Xi);

figure
subplot(3,1,1)
plot(t0:dt:tn, u, '-b'),grid
ylabel('control')
subplot(3,1,2)
plot(t0:dt:tn, x, '-b', t0:dt:tn, [x(1:D) cell2mat(Y)], '-r'), grid
ylabel('state')
subplot(3,1,3)
plot(t0+D*dt:dt:tn, x(D+1:end) - cell2mat(Y)), grid
ylabel('error')
xlabel('t')