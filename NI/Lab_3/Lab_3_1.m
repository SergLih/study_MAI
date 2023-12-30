%1.1 Генерация множества точек для каждой линии

[X1, Y1] = ellips(-0.2, 0, 0.2, 0.2, 0, 0.025);
[X2, Y2] = ellips(0, 0, 0.7, 0.5, -pi/3, 0.025);
[X3, Y3] = ellips(0, 0, 1, 1, 0, 0.025);

n = length(X1);
%Случайные точки 1-го класса (60 шт)
D1 = randperm(n);
n1 = 60;
D1 = D1(1:n1);
display(D1);
K1 = [ones(1, n1); 0*ones(1, n1); 0*ones(1, n1)];

%Случайные точки 2-го класса (100 шт)
D2 = randperm(n);
n2 = 100;
D2 = D2(1:n2);
display(D2);
K2 = [0*ones(1, n2); ones(1, n2); 0*ones(1, n2)];

%Случайные точки 3-го класса (120 шт)
D3 = randperm(n);
n3 = 120;
D3 = D3(1:120);
display(D3);
K3 = [0*ones(1, n3); 0*ones(1, n3); ones(1, n3)];

%1.2 Разделение точек на обучающее, контрольное, 
%и тестовое подмножества в отношении 70%-20%-10%
[trainInd1, valInd1, testInd1] = dividerand(length(D1), 0.7, 0.2, 0.1);
[trainInd2, valInd2, testInd2] = dividerand(length(D2), 0.7, 0.2, 0.1);
[trainInd3, valInd3, testInd3] = dividerand(length(D3), 0.7, 0.2, 0.1);

%1.3 Отображение исходных данных
figure

%Отображение 1-го класса
class1 = plot(X1, Y1, '-r', 'LineWidth', 2);

hold on;

tr1 = plot(X1(D1(trainInd1)), Y1(D1(trainInd1)), 'or', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'r', ...
    'MarkerSize', 7);

hold on;

val1 = plot(X1(D1(valInd1)), Y1(D1(valInd1)), 'rV', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'c', ...
    'MarkerSize', 7);

hold on;

test1 = plot(X1(D1(testInd1)), Y1(D1(testInd1)), 'rs', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'c', ...
    'MarkerSize', 7);

hold on;

%Отображение 2-го класса
class2 = plot(X2, Y2, '-g', ...
    'LineWidth', 2);

hold on;

tr2 = plot(X2(D2(trainInd2)), Y2(D2(trainInd2)), 'og', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'g', ...
    'MarkerSize', 7);

hold on;

val2 = plot(X2(D2(valInd2)), Y2(D2(valInd2)), 'gV', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'c', ...
    'MarkerSize', 7);

hold on;

test2 = plot(X2(D2(testInd2)), Y2(D2(testInd2)), 'gs', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'c', ...
    'MarkerSize', 7);

hold on;

%Отображение 3-го класса
class3 = plot(X3, Y3, '-b', ...
    'LineWidth', 2);

hold on;

tr3 = plot(X3(D3(trainInd3)), Y3(D3(trainInd3)), 'ob', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'b', ...
    'MarkerSize', 7);

hold on;

val3 = plot(X3(D3(valInd3)), Y3(D3(valInd3)), 'bV', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'c', ...
    'MarkerSize', 7);

hold on;

test3 = plot(X3(D3(testInd3)), Y3(D3(testInd3)), 'bs', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'c', ...
    'MarkerSize', 7);

hold on;

%1.4 Создание обучающей выборки
trainset = [trainInd1, trainInd2+60, trainInd3+160,...
    valInd1, valInd2+60, valInd3+160, ...
    testInd1, testInd2+60, testInd3+160];


%1.5 Создание сети
net = feedforwardnet(20);
net = configure(net, [-1.2 1.2; 0 1]);
net.layers{:}.transferFcn = 'tansig';
net.trainFcn = 'trainrp';

%1.6 Разделение обучающего множества на подмножества
net.divideFcn = 'divideind';

trnInd = length(trainInd1) + length(trainInd2) + length(trainInd3);
tstInd = length(valInd1) + length(valInd2) + length(valInd3);
proInd = length(testInd1) + length(testInd2) + length(testInd3);

net.divideParam.trainInd = 1:trnInd;
net.divideParam.valInd = (1:tstInd) + trnInd;
net.divideParam.testInd = (1:proInd) + (tstInd + trnInd);

%1.7 Инициализация
net = init(net);
%1.8 Задание параметров обучения
net.trainParam.epochs = 2500;
net.trainParam.max_fail = 1500;
net.trainParam.goal = 0.00001;

%1.9 Обучение сети
D = [D1, D2+length(X1), D3+length(X1)+length(X2)];
X = [X1, X2, X3];
Y = [Y1, Y2, Y3];
K = [K1, K2, K3];

[net, tr] = train(net, [X(D(trainset)); Y(D(trainset))], K(:,trainset));

%1.10 Отображение сети
display(net)

%1.11/12 Выход сети для обучающего, контрольного и тестового подмножеств, кол-во совпадений
trainInd = [trainInd1, trainInd2+60, trainInd3+160];
valInd = [valInd1, valInd2+60, valInd3+160];
testInd = [testInd1, testInd2+60, testInd3+160];

%Для обучающего
A = net([X(D(trainInd)); Y(D(trainInd))]);
nA = A >= 0.5;

fprintf('Размер обучающей выборки: %d\n',length(trainInd));
fprintf('Количество совпадений: %d\n\n',sum((sum(K(:,trainInd) == nA))==3));

%Для контрольного
A = net([X(D(valInd)); Y(D(valInd))]);
nA = A >= 0.5;

fprintf('Размер контрольной выборки: %d\n',length(valInd));
fprintf('Количество совпадений: %d\n\n',sum((sum(K(:,valInd) == nA))==3));

%Для тестового
A = net([X(D(testInd)); Y(D(testInd))]);
nA = A >= 0.5;

fprintf('Размер тестовой выборки: %d\n',length(testInd));
fprintf('Количество совпадений: %d\n\n',sum((sum(K(:,testInd) == nA))==3));

%1.13 Классификация точек области [-1.2,1.2]x[-1.2,1.2]
M = -1.2:0.010:1.2;
[gX, gY] = meshgrid(M);

A = net([gX(:)';gY(:)']);

%1.14 
mA = (round(10*A)/10)';
cmap = unique(mA,'rows');
out = [];

for i=1:length(mA)
    out = [out, find(ismember(cmap, mA(i,:), 'rows')==1)];
end;

out = reshape(out,length(gX),length(gY));
image(gX(:), gY(:),out);
colormap(cmap);



    