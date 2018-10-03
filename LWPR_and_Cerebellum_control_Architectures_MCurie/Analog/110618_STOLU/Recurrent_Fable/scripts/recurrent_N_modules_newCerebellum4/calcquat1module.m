
function [v] = calcquat1module(theta1, theta2, L1, L2, L3)
 % estimates the direct kinematics for robot ??? with two  modules connected.

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

 % Solving the direct kinematics for the robot - it must be done from joint 2 to BASE
 result1 = v1 + quatrotate(q2, v2);
 result2 = v0 + quatrotate(q1, result1);
 
 
 v = result2;
end