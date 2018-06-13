clear all;
clc;
A1 = 400;
A2 = 400;
phase = pi / 2;
dt = 0.01;
t0 = 0;
n_iter = 20000;
l1 = 0.15;
l2 = 0.045;
l3 = 0.21; %0.06;
l4 = 0.045; %0.21;
l5 = 0.06;



figure(3), hold on;

% Circle trajectory and plot 
for j=3:3
    for i=1:n_iter
        q1dd(i) = A1(j) * sin(2 * pi * t0);
        q1d(i) = ((-1/2)*pi) * A1(j) * cos(2 * pi * t0);
        q1(i) = (-power(((1/2)*pi),2)) * A1(j) * sin(2 * pi * t0);
        q2dd(i) = A2(j) * sin(2 * pi * t0 + phase);
        q2d(i) = ((-1/2)*pi) * A2(j) * cos(2 * pi * t0 + phase);
        q2(i) = (-power(((1/2)*pi),2)) * A2(j) * sin(2 * pi * t0 + phase);
        t0 = t0 + dt;
    end
    %plot(q1,q2);
    %plot(x,y);
end
traj = calcquat(q1, q2 ,q1 , q2, l1, l2, l3, l4, l5);
figure, plot3(traj(:,1), traj(:,2), traj(:,3));
old off;