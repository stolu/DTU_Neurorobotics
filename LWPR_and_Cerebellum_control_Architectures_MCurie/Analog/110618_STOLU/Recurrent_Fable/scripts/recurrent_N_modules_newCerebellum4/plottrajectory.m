clear all;
clc;
A1 = 7;
%A2 = 400;
phase = pi / 2;
dt = 0.01;
t0 = 0;
n_iter = 20000;
l1 = 0.145;
l2 = 0.055;
l3 = 0.22;  % 0.06;
l4 = 0.055;
l5 = 0.075;

% Eight trajectory and plot
for i = 1 : n_iter
    q1dd(i) = A1 * sin(2 * pi * t0);
    q1d(i) = ((-1/2)*pi) * A1 * cos(2 * pi * t0);
    q1(i) = (-power(((1/2)*pi),2)) * A1 * sin(2 * pi * t0);
    q2dd(i) = A1 * cos(4 * pi * t0 + phase);
    q2d(i) = ((1/2)*pi) * A1 * sin(4 * pi * t0 + phase);
    q2(i) = (-power(((1/2)*pi),2)) * A1 * cos(4 * pi * t0 + phase);
    t0 = t0 + dt;
end
%figure(2), plot(q1,q2); % joint space
 %plot(x,y);

% Cartesian space
%traj = calcquat1module(q1, q2, l1, l2, l3);
traj = calcquat(q1, q2, q1, q2, l1, l2, l3, l4, l5);

figure(3),hold on; plot(traj(:,1), traj(:,2));
hold off;

FileName = 'TestREC_1box_1modules_fab74_04-02-2018_11/52_15_0.0055_ltpm4ltdm3beta7p10m3BETTER.mat';

traj = calcquat(posr(:,1), posr(:,2), posr(), posr(), l1, l2, l3, l4, l5);
figure(4), hold on; plot(traj(:,1), traj(:,2));
hold off;

%%

for i = 1:n_iter
  %x(i) = l1 * sin(q1(i) * pi/180) + l2 * sin(q1(i)*pi/180) * sin(q2(i) * pi/180)';
  x(i) = l2 * sin(q1(i)*pi/180) * cos(q1(i)*pi/180) + l1 * sin(q1(i)*pi/180);
  %y(i) = l2 * sin(q2(i) * pi/180);
  y(i) = - l2 * sin(q2(i)*pi/180);
  %z(i) = (l1/10000) * cos(q1(i) * pi/180) + (l2/10000 ) * cos(q1(i) * pi/180) * cos(q2(i) * pi/180)';
  z(i) = l2 * cos(q1(i)*pi/180) * cos(q2(i)*pi/180) + l1 * cos(q1(i)*pi/180);
end

%x2 = l1 * sin(posr1 * pi/180) + l2 * sin(posr1 * pi/180) * sin(posr2 * pi/180)';
%y2 = l2 * sin(posr2 * pi/180);
%z2 = (l1/10000) * cos(posr1 * pi/180) + (l2 / 10000 ) * cos(posr1*pi/180) * cos(posr2 * pi/180)';


%figure(4),hold on; plot3(x,y,z);
hold off;