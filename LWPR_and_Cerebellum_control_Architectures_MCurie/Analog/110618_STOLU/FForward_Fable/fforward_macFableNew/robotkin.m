% q1 = 0:0.1:25;
% q2 = 0:0.1:25;
A = [700, 500, 900];
f = [0.5, 1, 0.75];
t = 0:0.001:1;
phase = pi / 2;
%q1 = -25*sin(2*pi*t);
%q2 = -25*sin(2*pi*t+pi/2);


l0 = 0.05;
l1 = 0.03;
l2 = 0.06;
L1 = Link('d', 0, 'a', l1, 'alpha', -pi/2);
L2 = Link('d', 0, 'a', l2, 'alpha', pi/2);
bot = SerialLink([L1 L2], 'name', 'my robot');

for j = 1:3
    q1 = -A(j)*power((1/(2*pi)),2)*sin(2*pi*f(1)*t);
    q2 = -A(j)*power((1/(2*pi)),2)*sin(2*pi*f(1)*t+phase);
    x = zeros(length(q1), 1);
    y = zeros(length(q1), 1);
    z = zeros(length(q1), 1);
    
    for i = 1:length(q1)
    
        eef = bot.fkine([q1(i)*pi/180 q2(i)*pi/180])*[0 0 0 1]';
        %     x(i) = eef(1);
        %     y(i) = eef(2);
        %     z(i) = eef(3);
        x(i) = eef(3);
        y(i) = eef(2);
        z(i) = eef(1)+l0;   
    end
    hold on;
    scatter3(x, y, z);
end
% scatter(x, y);

% bot.fkine([0.0 0.0])*[0 0 0 1]'
% bot.plot([0.0 0.0]);