
function [v] = calcquat(theta1, theta2, theta3, theta4, L1, L2, L3, L4, L5)
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