clc;
clear all;

% FileName = 'TestForward_05_04_2016_Mac.mat'; % base experiment

% FileName = 'TestForward_07_04_2016_Macweigth.mat'; % added a weight of 105.9 gr

% FileName = 'TestForward_01_04_2016_MaconlyPD.mat'; % Only a PD

% FileName = 'TestForward_velocity_Mac080416.mat'; % desired and real velocity in input to LWPR

% FileName = 'TestForward_08_04_2016_Macperturb.mat'; % hand perturbations 

% FileName = 'TestForward_11_04_2016_Mac_trj_switch.mat'; % Different trajectories

% FileName = 'TestForward_11_04_2016_Mac_eightFig.mat'; % Eight trajectory

% FileName = 'TestForward_14_04_2016_MacPerturb&load.mat'; % initial weight added (105.9 gr) and then hand perturbations

%FileName = 'TestForward_06_05_2016_Mac.mat';
%FileName = 'TestForward_06_05_2016_Mac_trj_switch.mat'; % Different trajectories
FileName = 'TestForward_06_05_2016_Mac_3freq.mat'; % Different frequencies
%FileName = 'TestForward_06_05_2016_Mac_loadI.mat'; % hand perturbations 
%FileName = 'TestForward_06_05_2016_onlypd.mat';
%FileName = 'TestForward_06_05_2016_onlypd_load.mat';
FileName = 'TestForward_ismaellombrice.mat';

FolderName = '/Users/stolu/Documents/CodeSilvia/lwpr_mycode_Jan2016/AFEL_Fable_LWPR/LWPRandFable/fforward_macFableNew/';
File = fullfile(FolderName, FileName);
load(File);

l8 = 1/0.01 + 1;
MSEep = [];
MAEep = [];
rtorqLF = [];
rCtorq = [];
rLWPRtorq = [];
N = length(q1);

for i = 1 : N
 rtorqLF(1,i) = abs(torquesLF1(1,i)) ./ (abs(torquesLF1(1,i)) + abs(torqLWPR1(1,i)) + abs(Ctorques1(1,i)));
 rCtorq(1,i) = abs(Ctorques1(1,i)) ./ (abs(torquesLF1(1,i)) + abs(torqLWPR1(1,i)) + abs(Ctorques1(1,i)));
 rLWPRtorq(1,i) = abs(torqLWPR1(1,i)) ./ (abs(torquesLF1(1,i)) + abs(torqLWPR1(1,i)) + abs(Ctorques1(1,i)));
 
 rtorqLF(2,i) = abs(torquesLF2(1,i)) ./ (abs(torquesLF2(1,i)) + abs(torqLWPR2(1,i)) + abs(Ctorques2(1,i)));
 rCtorq(2,i) = abs(Ctorques2(1,i)) ./ (abs(torquesLF2(1,i)) + abs(torqLWPR2(1,i)) + abs(Ctorques2(1,i)));
 rLWPRtorq(2,i) = abs(torqLWPR2(1,i)) ./ (abs(torquesLF2(1,i)) + abs(torqLWPR2(1,i)) + abs(Ctorques2(1,i)));
 
 if ~mod(i,l8)
   pinit = -l8 + (i + 1);
   pfin = pinit + (l8 - 1);
   n = round(i / l8);
   i = i + 1;
   MSEep(n,1) = mse((q1(1,pinit:pfin) - posr1(1,pinit:pfin)));
   MSEep(n,2) = mse((q2(1,pinit:pfin) - posr2(1,pinit:pfin)));
   MAEep(n,1) = mse((q1(1,pinit:pfin) - posr1(1,pinit:pfin)));
   MAEep(n,2) = mse((q2(1,pinit:pfin) - posr2(1,pinit:pfin)));
   nMSEep(n,1) = MSEep(n,1)./var(q1(1,pinit:pfin));
   nMSEep(n,2) = MSEep(n,2)./var(q2(1,pinit:pfin));
   
   rLFm(n,1)= mean(rtorqLF(1,pinit:pfin));
   rLFm(n,2)= mean(rtorqLF(2,pinit:pfin));
   rCm(n,1)= mean(rCtorq(1,pinit:pfin));
   rCm(n,2)= mean(rCtorq(2,pinit:pfin));
   rLWPRm(n,1)= mean(rLWPRtorq(1,pinit:pfin));
   rLWPRm(n,2)= mean(rLWPRtorq(2,pinit:pfin));
 end
 
end
rLFmj = mean(rLFm,2);
rCmj = mean(rCm,2);
rLWPRmj = mean(rLWPRm,2);

% Plot circle figure
figure(8);
plot(q1(1, 1:l8), q2(1, 1:l8));
%plot(x(1, 1:l8), y(1, 1:l8));
t = 0:0.01:1;
figure(9); hold on
plot(t, q1(1, 1:l8));
plot(t, q2(1, 1:l8));
hold off;

%figure(10); hold on
%plot(q1d(1, 1:end));
%plot(velr1(1, 1:end));
%legend('q1d','');
%hold off;

% Plot MSE 
figure(1), hold on;
plot(nMSEep(:,1), 'r');
plot(nMSEep(:,2), 'b');
legend('nMSE joint 1','nMSE joint 2');
xlabel('N. iterations');
ylabel('nMSE (degrees)');
hold off;

% Plot MAE 
%figure(2), hold on;
%plot(MAEep(:,1), 'r');
%plot(MAEep(:,2), 'b');
%hold off;

% Plot Torques
figure(3), hold on;
plot(Ctorques1, 'r');
%plot(Ctorques2, 'r');
plot(torqLWPR1, 'g');
%plot(torqLWPR2, 'g');
plot(torquesLF1, 'b');
%plot(torquesLF2, 'b');
plot(torquestot1, 'k');
%plot(torquestot2, 'k');
legend('Ctorques1','torqLWPR1','torquesLF1','torquestot1');
xlabel('N. iterations');
hold off;

% Plot mean torques joint 1 
%figure(4), hold on;
%plot(rLFm(:,1), 'r');
%plot(rLWPRm(:,1), 'k');
%plot(rCm(:,1), 'g');
%legend('Averaged torque LF','Averaged torque LWPR', 'Averaged torque Cerebellum');
%xlabel('N. iterations');
%ylabel('nMSE torque joint 1');
%hold off;

% Plot mean torques joint 2
%figure(5), hold on;
%plot(rLFm(:,2), 'r');
%plot(rLWPRm(:,2), 'k');
%plot(rCm(:,2), 'g');
%legend('Averaged torque LF','Averaged torque LWPR', 'Averaged torque Cerebellum');
%xlabel('N. iterations');
%ylabel('nMSE torque joint 2');
%hold off;

% Plot mean torques between joints
figure(6), hold on;
plot(rLFmj, 'r');
plot(rLWPRmj, 'k');
plot(rCmj, 'g');
legend('Averaged torque LF','Averaged torque LWPR', 'Averaged torque Cerebellum');
xlabel('N. iterations');
ylabel('Averaged torques among joints');
hold off;

