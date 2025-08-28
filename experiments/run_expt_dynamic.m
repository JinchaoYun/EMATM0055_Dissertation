%% =========================
%  Experiment B (revised): 动态正弦扰动的影响（单模型跨A泛化）
%  - 仅在 A = 0 (无动态扰动) 上训练一次
%  - 固定训练得到的模型与归一化参数 normp
%  - 在 A ∈ {0, 0.1, 0.3} deg 下仅做推理评估
% ==========================
clear; clc; close all;
addpath(genpath('../src'));

C = config();
C.T_sec           = 5;
C.sample_hz       = 200;            % 固定采样频率
C.static_bias_deg = 0.5;            % 固定静态偏差
C.f_dyn_hz        = 10.0;            % 固定动态频率（控制变量）
C.save_tag        = sprintf('DYN_fdyn%.1fHz', C.f_dyn_hz);

% ---------- 1) 只在 A = 0 上训练一次 ----------
C_train = C;
C_train.use_dynamic = false;        % 训练时不加动态扰动
[Xtr, Ytr, ~] = build_dataset(C_train);
[dsTrain, dsVal, dsTest, normp] = split_normalize(Xtr, Ytr, C_train);

Rtrain = bpnn_train_eval(dsTrain, dsVal, dsTest, normp, C_train, 'TRAIN_A0');
netTrained = Rtrain.model;          % 取出已训练模型
fprintf('Baseline (A=0) — Acc=%.3f, F1(m)=%.3f, CE=%.4f\n', ...
    Rtrain.acc_overall, (2*Rtrain.prec_macro*Rtrain.recall_macro)/(Rtrain.prec_macro+Rtrain.recall_macro+eps), Rtrain.loss_xent);

% ---------- 2) 固定模型，跨 A 做评估 ----------
A_list_deg = [0, 0.1, 0.3];
out_all = [];

for A = A_list_deg
    C_eval = C;
    C_eval.use_dynamic = true;      % 评估时开启动态扰动
    C_eval.A_dyn_deg   = A;

    % 生成整套评估数据（不再划分 train/val/test；整套作为 test）
    [Xe, Ye, ~] = build_dataset(C_eval);    % Xe: [N x D], Ye: [N x K]

    % 用训练得到的 normp 归一化（与训练一致，避免泄漏评估分布）
    xmin = normp.xmin; xmax = normp.xmax; eps_ = normp.eps;
    Xe_n = (Xe - xmin) ./ max(xmax - xmin, eps_);
    Xe_n = min(max(Xe_n, 0), 1);

    % 推理
    Yhat = netTrained(Xe_n');                 % [K x N]
    % 指标
    [acc, prec, rec, cm] = compute_metrics(Ye, Yhat');    % 传 [N x K]
    ce  = -mean(sum(Ye .* log(max(Yhat',1e-12)), 2));     % 交叉熵
    [~,~,~,auc_macro, hROC] = helpers.plot_roc_ova(Ye, Yhat');

    % 保存图（混淆矩阵 + ROC）
    if ~exist('../figures','dir'); mkdir('../figures'); end
    hCM = figure('Name', sprintf('Confusion_A=%.1fdeg_f%gHz', A, C.f_dyn_hz));
    plotconfusion(Ye', Yhat);  % [K x N]
    saveas(hCM, fullfile('../figures', sprintf('confusion_dyn_A%.1fdeg_f%gHz.png', A, C.f_dyn_hz)));
    saveas(hROC, fullfile('../figures', sprintf('roc_dyn_A%.1fdeg_f%gHz.png', A, C.f_dyn_hz)));
    close(hCM); close(hROC);

    % 记录
    out_all = [out_all; {A, acc, ce, mean(prec), mean(rec), auc_macro}]; %#ok<AGROW>

    fprintf('[A=%.3f deg] Acc=%.3f  CE=%.4f  Prec(m)=%.3f  Rec(m)=%.3f  AUC(m)=%.3f\n', ...
        A, acc, ce, mean(prec), mean(rec), auc_macro);
end

T = cell2table(out_all, 'VariableNames', ...
    {'A_dyn_deg','accuracy','cross_entropy','precision_macro','recall_macro','AUC_macro'});
disp(T);

if ~exist('../results','dir'); mkdir('../results'); end
writetable(T, fullfile('../results', sprintf('summary_%s.csv', C.save_tag)));
save(fullfile('../results', sprintf('summary_%s.mat', C.save_tag)), 'T', 'Rtrain');
