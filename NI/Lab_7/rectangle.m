function [ o ] = rectangle( t, d1, d2, x0, y0, alpha, eps)
if nargin < 4
    x0 = 0;
    y0 = 0;
    alpha = 0;
end
beta = atan(d2/d1);
phi = mod(t + alpha, 2*pi)
% if t == 2.8250
%     print("aaa")
% end
if t == 0 || abs(t - pi) < eps
    r = d1/2;
elseif abs(t - pi/2) < eps || abs(t - 3*pi/2) < eps
    r = d2/2;
elseif (pi+beta < t && t < 2*pi-beta) || (beta < t && t < pi-beta)
    r = abs(d2 / (2 * sin(t)));
else%if (0 < t && t < beta) || (2*pi-beta < t && t < 2*pi) || (pi-beta < t && t < pi+beta)
    r = abs(d1 / (2 * cos(t)));
end
o = [r * cos(phi)+x0; r*sin(phi)+y0]
%x1 = a * cos(t); 
%y1 = b * sin(t);
%o = [x1 * cos(alpha) - y1 * sin(alpha) + x0;
%     x1 * sin(alpha) + y1 * cos(alpha) + y0];
end

