function DH = ur10_dh(theta_deg)
% 返回 6×4 的 DH 表（[a, alpha(deg), d, theta(deg)]），theta 由输入设置
% UR10（标准 DH，一致于你论文表）
a     = [   0,  -612.0,  -572.3,    0,    0,   0];
alpha = [  90,     0,       0,     90,  -90,   0];
d     = [127.3,    0,       0,   163.9, 115.7, 92.2];
theta = theta_deg(:).';

DH = [a(:), alpha(:), d(:), theta(:)];
end
