%Использование обобщенно-регрессионной нейронной сети для апроксимации
%функции

t0 = 0;
tn = 5;
h = 0.025;
n = (tn - t0) / h + 1;

t = t0:h:tn;
x = sin(t.*t - 2*t + 5);

%3.1-3.4 Создание сети
spread = h;

[trainInd, testInd] = dividerand(n, .8, .2);
net = newgrnn(t, x, spread);
view(net);

y = sim(net, t);
fprintf('Train error: %d\n',sqrt(mse(x(trainInd) - y(trainInd))));
fprintf('Test error: %d\n',sqrt(mse(x(testInd) - y(testInd))));

%3.5-3.8 Графики
figure
referenceLine = plot(t, x, 'r');

hold on;
approximationLine = plot(t, y, '-b');

grid on;
legend([referenceLine,approximationLine],'reference line', 'approximation line');
figure;
plot(t, x - y, 'r');
grid on;

net = newgrnn(t(trainInd), x(trainInd), spread);
y = sim(net, t);
figure
referenceLine = plot(t, x, 'r');
set(referenceLine, 'linewidth', 4);

hold on;
approximationLine = plot(t, y, '--b');
set(approximationLine, 'linewidth', 4);

grid on;
legend([referenceLine,approximationLine],'reference line', 'approximation line');
figure;
plot(t, x - y, 'r');
grid on;