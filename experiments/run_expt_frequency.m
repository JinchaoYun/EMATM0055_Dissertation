%% =========================
%  Experiment A: 采样频率影响
%  - 固定轨迹时长 T=5 s
%  - 频率 sweep: 50/200/800 Hz → N = 250 / 1000 / 4000
%  - 固定静态偏差（0.1/0.5/1.0 deg），不叠加动态扰动
% ==========================
clear; clc; close all;
addpath(genpath('../src'));

C = config();                        % 全局配置（UR10、起止姿态、网络结构等）
C.T_sec           = 5;
C.freq_list_hz    = [50, 200, 800];
C.static_bias_deg = 0.5;             % 固定静态偏差
C.use_dynamic     = false;           % 不加正弦扰动
C.save_tag        = sprintf('%g_DEGR_FREQ_ONLY', C.static_bias_deg);

out_all = [];                        % 汇总表

for f = C.freq_list_hz
    C.sample_hz = f;

    % 1) 构建数据集（7 类：0=无故障, 1..6=关节故障）
    [X, Y, meta] = build_dataset(C);

    % 2) 划分并归一化（仅用训练集计算全局 min/max）
    [dsTrain, dsVal, dsTest, normp] = split_normalize(X, Y, C);

    % 3) 训练 + 评估（显示训练窗口 & 生成图）
    R = bpnn_train_eval(dsTrain, dsVal, dsTest, normp, C, sprintf('%g_DEGR_FREQ_%dhz', C.static_bias_deg, f));

    % 4) 收集结果
    out_all = [out_all; {f, R.acc_overall, R.loss_xent, R.prec_macro, R.recall_macro, R.auc_macro}]; %#ok<AGROW>
end

% 5) 输出汇总
T = cell2table(out_all, 'VariableNames', ...
    {'freq_Hz','accuracy','cross_entropy','precision_macro','recall_macro','AUC_macro'});
disp(T);

if ~exist('../results','dir'); mkdir('../results'); end
writetable(T, fullfile('../results', sprintf('summary_%s.csv', C.save_tag)));
save(fullfile('../results', sprintf('summary_%s.mat', C.save_tag)), 'T');
