function Theta_fault = inject_joint_fault(Theta_nom, C, joint_id)
% 在 joint_id (1..6) 上注入：固定偏差 + 可选正弦扰动；其余保持 nominal
Theta_fault = Theta_nom;
if joint_id == 0          % 0 = 无故障
    return;
end
bias = C.static_bias_deg; % 固定偏差
Theta_fault(:, joint_id) = Theta_fault(:, joint_id) + bias;

if C.use_dynamic && C.A_dyn_deg > 0
    N = size(Theta_fault,1);
    t = linspace(0, C.T_sec, N).';
    Theta_fault(:, joint_id) = Theta_fault(:, joint_id) + C.A_dyn_deg * sin(2*pi*C.f_dyn_hz*t);
end
end
