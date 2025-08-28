function [X, Y_onehot, meta] = build_dataset(C)
% 生成 7 类数据（normal + 6 joints fault），特征为 [x,y,z,roll,pitch,yaw]
[Theta_nom, ~] = plan_joint_trajectory(C);
N = size(Theta_nom,1);

X = [];  Y_onehot = [];
for cls = 0:6
    if cls == 0
        Theta = Theta_nom;
    else
        Theta = inject_joint_fault(Theta_nom, C, cls);
    end

    % 逐样本做正运动学 → 位置 + 欧拉
    feats = zeros(N, 6);
    for k = 1:N
        T06 = fwd_kinematics_ur10(Theta(k,:));
        p   = T06(1:3,4).';           % [x y z]
        R   = T06(1:3,1:3);
        eul = euler_zyx_from_R(R);    % [roll pitch yaw]（deg）
        feats(k,:) = [p, eul];
    end

    X = [X; feats]; %#ok<AGROW>

    lab = zeros(1,7); lab(cls+1) = 1;       % one-hot
    Y_onehot = [Y_onehot; repmat(lab, N, 1)]; %#ok<AGROW>
end

meta.N_per_class = N;
meta.labels = C.class_names(:);
end
