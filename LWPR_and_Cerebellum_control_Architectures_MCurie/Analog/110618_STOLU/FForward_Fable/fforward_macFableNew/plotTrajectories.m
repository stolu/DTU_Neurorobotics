clear all;
clc;
A1 = [10, 5, 20, 5, 20, 10, 5, 10, 20, 5, 20, 10, 5, 10, 20];
A2 = [10, 5, 20, 5, 20, 10, 5, 10, 20, 5, 20, 10, 5, 10, 20];
A3 = 20;
phase = pi / 2;
dt = 0.01;
t0 = 0;
n_iter = 20000;
l1 = 0.0308;
l2 = 0.088;

figure(3), hold on;

% Circle trajectory and plot 
for j=1:3
    for i=1:n_iter
        q1dd(i) = A1(j) * sin(2 * pi * t0);
        q1d(i) = ((-1/2)*pi) * A1(j) * cos(2 * pi * t0);
        q1(i) = (-power(((1/2)*pi),2)) * A1(j) * sin(2 * pi * t0);
        q2dd(i) = A2(j) * sin(2 * pi * t0 + phase);
        q2d(i) = ((-1/2)*pi) * A2(j) * cos(2 * pi * t0 + phase);
        q2(i) = (-power(((1/2)*pi),2)) * A2(j) * sin(2 * pi * t0 + phase);
        t0 = t0+dt;
        % Cartesian coordinates
        x(i) = l1 * cos(q1(i)*pi/180) + l2 * cos((q1(i) + q2(i))*pi/180);
        y(i) = l1 * sin(q1(i)*pi/180) + l2 * sin((q1(i) + q2(i))*pi/180);
    end
    plot(q1,q2);
    %plot(x,y);
end
hold off;

%eight trajectory and plot
t0 = 0;
figure(4), hold on;

for i=1:n_iter
    q1(i) = sin(2*pi*t0) * A3;
    q2(i) = cos(2*pi*t0 + pi/4) * A3;
    
    q1d(i) = cos(2*pi*t0) * (A3 * 2 * pi);
    q2d(i) = -sin(2*pi*t0 + pi/4)* (A3 * 2 * pi);
    
    q1dd(i) = (-sin(2*pi*t0)* (A3 * 4 * power(pi,2)));
    q2dd(i) = (-cos(2*pi*t0 + pi/4)* (A3 * 4 * power(pi,2)));
    t0 = t0+dt;
    %Cartesian coordinates
    x(i) = l1 * cos(q1(i)*pi/180) * (l2 * cos(q2(i)*pi/180) + cos(q2(i)*pi/180));
    y(i) = l1 * sin(q1(i)*pi/180) * (l2 * cos(q2(i)*pi/180) + cos(q2(i)*pi/180));
end
%plot(q1,q2);
plot(x,y);
hold off;
