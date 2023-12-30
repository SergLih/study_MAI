function [X, Y] = ellips(xc, yc, a, b, phi, h)
	t = 0:h:2*pi;
	X = xc + a*cos(t)*cos(phi) - b*sin(t)*sin(phi);
	Y = yc + b*cos(t)*sin(phi) + a*sin(t)*cos(phi);