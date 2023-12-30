%Использование сети с радиальными базисными элементами для классификации
%точек в случае, когда классы не являются линейно разделимыми

%2.1 Генерация множества точек для каждой линии

[X1, Y1] = ellips(-0.2, 0, 0.2, 0.2, 0, 0.025);
[X2, Y2] = ellips(0, 0, 0.7, 0.5, -pi/3, 0.025);
[X3, Y3] = ellips(0, 0, 1, 1, 0, 0.025);

n = length(X1);
%Случайные точки 1-го класса (60 шт)
D1 = randperm(n);
n1 = 60;
D1 = D1(1:n1);
K1 = [ones(1, n1); 0*ones(1, n1); 0*ones(1, n1)];

%Случайные точки 2-го класса (100 шт)
D2 = randperm(n);
n2 = 100;
D2 = D2(1:n2);
K2 = [0*ones(1, n2); ones(1, n2); 0*ones(1, n2)];

%Случайные точки 3-го класса (120 шт)
D3 = randperm(n);
n3 = 120;
D3 = D3(1:120);
K3 = [0*ones(1, n3); 0*ones(1, n3); ones(1, n3)];

%2.2 Разделение точек на обучающее
%и тестовое подмножества в отношении 80%-20%
[trainInd1, testInd1] = dividerand(length(D1), 0.8, 0.2);
[trainInd2, testInd2] = dividerand(length(D2), 0.8, 0.2);
[trainInd3, testInd3] = dividerand(length(D3), 0.8, 0.2);

%2.3 Отображение исходных данных
figure
hold on;

%Отображение 1-го класса
class1 = plot(X1, Y1, '-r', ...
    'LineWidth', 2);

tr1 = plot(X1(D1(trainInd1)), Y1(D1(trainInd1)), 'or', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'r', ...
    'MarkerSize', 7);

test1 = plot(X1(D1(testInd1)), Y1(D1(testInd1)), 'rs', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'r', ...
    'MarkerSize', 7);

%Отображение 2-го класса
class2 = plot(X2, Y2, '-g', ...
    'LineWidth', 2);

tr2 = plot(X2(D2(trainInd2)), Y2(D2(trainInd2)), 'og', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'g', ...
    'MarkerSize', 7);

test2 = plot(X2(D2(testInd2)), Y2(D2(testInd2)), 'gs', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'g', ...
    'MarkerSize', 7);

%Отображение 3-го класса
class3 = plot(X3, Y3, '-b', ...
    'LineWidth', 2);

tr3 = plot(X3(D3(trainInd3)), Y3(D3(trainInd3)), 'ob', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'b', ...
    'MarkerSize', 7);

test3 = plot(X3(D3(testInd3)), Y3(D3(testInd3)), 'bs', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'b', ...
    'MarkerSize', 7);

legend('Исходное множество 1', 'Обучающее подмножество 1', 'Тестовое подмножество 1',...
    'Исходное множество 2', 'Обучающее подмножество 2', 'Тестовое подмножество 2',...
    'Исходное множество 3', 'Обучающее подмножество 3', 'Тестовое подмножество 3');



%2.4 Обьединение соответствующега подмножества точек в обучающее и тестовое
trainset = [trainInd1, trainInd2+60, trainInd3+160];
testset = [testInd1, testInd2+60, testInd3+160];
X = [X1, X2, X3];
Y = [Y1, Y2, Y3];
D = [D1, D2+length(X1), D3+length(X1)+length(X2)];
K = [K1, K2, K3];

%2.5 Создание сети
spread = 0.3;
goal = 1.0e-5;
net = newrb([X(D(trainset)); Y(D(trainset))], K(:,trainset), goal, spread);

%2.6 Отображение структуры сети
view(net);

%2.7/2.8 Проверка качества
PK = net([X(D(trainset)); Y(D(trainset))]);
TK = K(:,trainset);

fprintf('Размер обучающей выборки: %d\n',length(trainset));
fprintf('Количество совпадений: %d\n\n',sum(TK(TK == PK)));

PK = net([X(D(testset)); Y(D(testset))]);
TK = K(:,testset);

fprintf('Размер тестовой выборки: %d\n',length(testset));
fprintf('Количество совпадений: %d\n\n',sum(TK(TK == PK)));

%2.9 Классификация точек области
figure
hold on;
axis([-1.2 1.2 -1.2 1.2]);
grid on

[gX, gY] = meshgrid(-1.2:0.025:1.2, 1.2:-0.025:-1.2);

A = net([gX(:)';gY(:)']);
n = length(gX);
A = max(0, min(1, A));
A = round(A * 10) * 0.1;

ctable = unique(A', 'rows');
cmap = zeros(n, n);

for i = 1 : size(ctable, 1)
    cmap(ismember(A', ctable(i, :), 'rows')) = i; 
end
image([-1.2, 1.2], [-1.2, 1.2], cmap); 
colormap(ctable);


hold on;
axis([-1.2 1.2 -1.2 1.2]);
grid on

%2.10 Меняем spread
spread = 0.1;
goal = 1.0e-5;
net = newrb([X(D(trainset)); Y(D(trainset))], K(:,trainset), goal, spread);
view(net);

%2.11 Классификация точек области
A = net([gX(:)';gY(:)']);
n = length(gX);
A = max(0, min(1, A));
A = round(A * 10) * 0.1;

ctable = unique(A', 'rows');
cmap = zeros(n, n);

for i = 1 : size(ctable, 1)
    cmap(ismember(A', ctable(i, :), 'rows')) = i; 
end
image([-1.2, 1.2], [-1.2, 1.2], cmap); 
colormap(ctable);

