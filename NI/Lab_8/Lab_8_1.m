% 1.1 - 1.2 считываем данные из файла
fileID = fopen('SN_m_190203_now.txt','r');
formatSpec = '%f';
sizeData = [1 Inf];
wolf_dataset = fscanf(fileID,formatSpec,sizeData);
fclose(fileID);

% 1.3 сглаживание
x = wolf_dataset;
x = smooth(x, 12);
%plot(x, '-r');
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
title('выход сети на обучающем множестве')

%%

figure;
hold on;
grid on;
plot([cell2mat(Xi) cell2mat(Y)] - cell2mat(x), '-r');
title('ошибка обучения на обучающем множестве')
% %
xm = cell2mat(x);
ym = cell2mat(Y);

figure;
hold on;
grid on;
plot(xm(ntrain + nval + 1 : ntrain + nval + ntest), '-b');
plot(ym(ntrain + nval - 9 : ntrain + nval + ntest - 10), '-r');
title('')