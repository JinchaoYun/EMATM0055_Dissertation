function [Theta_deg, t] = plan_joint_trajectory(C)
% 简单的关节空间线性插值（起止在 config 中给出），共 N = T*fs 个采样
N = round(C.T_sec * C.sample_hz);
t = linspace(0, C.T_sec, N).';
q0 = C.joint_start_deg(:).';
q1 = C.joint_end_deg(:).';
Theta_deg = q0 + (q1 - q0) .* (t / C.T_sec);  % 每列一个关节
end
