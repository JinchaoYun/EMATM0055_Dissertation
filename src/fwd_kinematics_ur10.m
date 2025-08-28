function T06 = fwd_kinematics_ur10(theta_deg)
% 基于 DH 的正运动学，输出 4×4 齐次变换矩阵
DH = ur10_dh(theta_deg);
T06 = eye(4);
for i = 1:6
    a = DH(i,1);  alpha = DH(i,2)*pi/180;  d = DH(i,3);  th = DH(i,4)*pi/180;
    A = [ cos(th),           -sin(th)*cos(alpha),  sin(th)*sin(alpha),  a*cos(th);
          sin(th),            cos(th)*cos(alpha), -cos(th)*sin(alpha),  a*sin(th);
                0,                    sin(alpha),          cos(alpha),          d;
                0,                             0,                   0,          1];
    T06 = T06 * A;
end
end
