clc;
clear all;

FolderName = '/Users/stolu/Documents/CodeSilvia/lwpr_mycode_Jan2016/AFEL_Fable_LWPR/LWPRandFable/recurrent_macFableNEW/2modules';
FileName = 'TestREC_1box_2modules_fab74_07-09-2017_10:43_14_0.0055.mat'; % 2013 Tolu's work - 1 module
%FileName = 'TestREC_1box_2modules_fab74_07-09-2017_10:27_17_0.0055.mat'; %added second module later
%FileName = 'TestREC_1box_2modules_fab_noPCDCN80_07-09-2017_11:07_18_0.0055.mat';
File = fullfile(FolderName, FileName);
load(File);

% Inizialitation variables
l1 = 0.15;
l2 = 0.045;
l3 = 0.21;  %0.06;
l4 = 0.045; %0.21;
l5 = 0.06;
dt = 0.01;
l8 = 1/dt + 1;

N = length(q1);
MAEepmod1 = [];
MSEepmod1 = [];
nMSEepmod1 = [];
nMSEeptmod1 = [];
MSEepmod2 = [];
nMSEepmod2 = [];
nMSEeptmod2 = [];

% Real pos module 1
posr0mod1 = posr0;
posr1mod1 = posr1;
errp0mod1 = errp0;
errp1mod1 = errp1;

FileName = 'TestREC_1box_2modules_fab80_07-09-2017_10:43_14_0.0055.mat'; % prediction errors and 2013 tolu's work - 1 module
%FileName = 'TestREC_1box_2modules_fab80_07-09-2017_10:27_17_0.0055.mat'; % added second joint later
%FileName = 'TestREC_1box_2modules_fab_noPCDCN74_07-09-2017_11:07_18_0.0055.mat';
File = fullfile(FolderName, FileName);
load(File);
% Real pos module 2
posr0mod2 = posr0;
posr1mod2 = posr1;
errp0mod2 = errp0;
errp1mod2 = errp1;

for i = 1 : N
 if ~mod(i,l8)
   pinit = -l8 + (i + 1);
   pfin = pinit + (l8 - 1);
   r = round(i / l8);
   MSEepmod1(r,1) = mse(q0(1,pinit:pfin) - posr0mod1(1,pinit:pfin));%errp1mod1(pinit:pfin));%));
   MSEepmod1(r,2) = mse(q1(1,pinit:pfin) - posr1mod1(1,pinit:pfin));%errp2mod1(pinit:pfin));%(
   MSEepmod2(r,1) = mse(q0(1,pinit:pfin) - posr0mod2(1,pinit:pfin));%errp1mod2(pinit:pfin));%
   MSEepmod2(r,2) = mse(q1(1,pinit:pfin) - posr1mod2(1,pinit:pfin));%errp2mod2(pinit:pfin));%
   
   MAEepmod1(r,1) = mae((q0(1,pinit:pfin) - posr0mod1(1,pinit:pfin)));%./ var(q1(1,pinit:pfin));
   MAEepmod1(r,2) = mae((q1(1,pinit:pfin) - posr1mod1(1,pinit:pfin)));%./ var(q2(1,pinit:pfin));
   
   nMSEepmod1(r,1) = MSEepmod1(r,1)./ var(q0(1,pinit:pfin));
   nMSEepmod1(r,2) = MSEepmod1(r,2)./ var(q1(1,pinit:pfin));
   nMSEeptmod1(r,1) = smooth(mean(nMSEepmod1(r,1:2)));
   
   nMSEepmod2(r,1) = MSEepmod2(r,1)./ var(q0(1,pinit:pfin));
   nMSEepmod2(r,2) = MSEepmod2(r,2)./ var(q1(1,pinit:pfin));
   nMSEeptmod2(r,1) = smooth(mean(nMSEepmod2(r,1:2)));
 end
end

% Plot nMSE for two modules averaged among joints 
figure(1), hold on;
plot(nMSEeptmod1(:,:), 'r');
plot(nMSEeptmod2(:,:), 'b');
%plot(MAEepmod1,'g'); % plot mae module 1
xlabel('N. iterations');
ylabel('Averaged nMSE of each robot module among joints');
legend('Averaged nMSE Module 1','Averaged nMSE Module 2');
hold off;

% Plot desired figure
traj = calcquat(q0(1,1:l8),q1(1,1:l8),q0(1,1:l8),q1(1,1:l8), l1,l2,l3,l4,l5);
figure(9), hold on;
plot3(traj(:,1), traj(:,2), traj(:,3), 'r');

% Plot real figure outcome - outcome n. 5
traj = calcquat(posr0mod1(1,l8*18:l8*18+l8),posr1mod1(1,l8*18:l8*18+l8),posr0mod2(1,l8*18:l8*18+l8),posr1mod2(1,l8*18:l8*18+l8), l1,l2,l3,l4,l5); 
plot3(traj(:,1), traj(:,2), traj(:,3),'b');

% Plot last 5 outcomes
a = (size(q0,2)-l8*5);
for h = a : size(q1,2)-l8 %(2001)   
 if ~mod(h,l8)
   pinit = h;
   pfin = pinit + l8;   
   traj = calcquat(posr0mod1(1, pinit:pfin),posr1mod1(1,pinit:pfin),posr0mod2(1,pinit:pfin),posr1mod2(1,pinit:pfin), l1,l2,l3,l4,l5);
   plot3(traj(:,1), traj(:,2), traj(:,3),'color',rand(1,3));
 end
end
%hold off;

% Plot MAE 
%figure(2), hold on;
%plot(MAEepmod1(:,1), 'r');
%plot(MAEepmod1(:,2), 'b');
%hold off;

figure(3), hold on;
plot(posr0mod1, 'r');
plot(poslwpr1, 'b');
plot(DCNp1,'g');
plot(posC1,'g');
legend('Real pos','Lwpr pos', 'Cerebellar pos');
hold off;

figure(4), hold on;
plot(posr0mod1, 'r');
plot(vellwpr1, 'b');
plot(q1d, 'g');
plot(velC1,'m');
legend('Real vel','Lwpr vel', 'Desired vel', 'Cerebellar vel');
hold off;

%figure(5), hold on;
%plot(errp1, 'r');
%plot(poslwpr1, 'b');
%legend('trajectory error 1','Lwpr errp1');
%hold off;

function v = calcquat(theta1, theta2, theta3, theta4, L1, L2, L3, L4, L5)
% estimates the direct kinematics for robot ??? with two
% modules connected.

% Check for theta as a vector - it will help to estimate a complete angular
% trajectory
s = size(theta1);
if(s(2)==1)
    theta1 = theta1';
    theta2 = theta2';
    s = size(theta1);
end

% Module 1
v0 = [0, 0, L1];
v1 = [0, L2, 0]; 
v2 = [L3, 0, 0];
% Module 2
v3 = [0 L4 0];
v4 = [L5 0 0];

% Quaternion from joint 1 to BASE
qx = [cosd(theta1/2)' zeros(s(2),1) zeros(s(2),1) sind(theta1/2)'];
qxx = [cosd(-45)' sind(-45)' 0 0];
q1 = quatmultiply(qx,qxx);
% Quaternion from joint 2 to joint 1
qx = [cosd(theta2/2)' zeros(s(2),1) zeros(s(2),1) sind(theta2/2)'];
qz = [cosd(45)' 0 0 sind(45)'];
qy = [cosd(-45)' 0 sind(-45)' 0];
q2 = quatmultiply(qx, quatconj(qz));
q2 = quatmultiply(q2, qy);
% Quaternion from joint 3 to joint 2
qx = [cosd(theta3/2)' zeros(s(2),1) zeros(s(2),1) sind(theta3/2)'];
q3 = quatmultiply(qx, quatconj(qy));
q3 = quatmultiply(q3, (qz));
% Quaternion from joint 4 to joint 3
qx = [cosd(theta4/2)' zeros(s(2),1) zeros(s(2),1) sind(theta4/2)'];
q4 = quatmultiply(qx, quatconj(qz));
q4 = quatmultiply(q4, qy);

% Solving the direct kinematics for the robot - it must be done from joint 4 to BASE
result1 = v3 + quatrotate(q4, v4);
result2 = v2 + quatrotate(q3, result1);
result3 = v1 + quatrotate(q2, result2);
result4 = v0 + quatrotate(q1, result3);

v = result4;
end
