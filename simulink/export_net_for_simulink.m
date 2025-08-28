function export_net_for_simulink()
% 导出 200 Hz 训练得到的网络为可代码生成的 predict_bpn.m，
% 同时把归一化参数保存为 params_norm.mat，供 Simulink 常量块读取。
clc;
% 1) 定位并加载 200 Hz 实验的 artifact
art = [];
candidates = dir(fullfile('..','results','artifact_0.5*200*hz*.mat'));
if isempty(candidates)
    candidates = dir(fullfile('..','results','artifact_0.5_DEGR_FREQ_200hz.mat'));
end
assert(~isempty(candidates), '未找到 200Hz 的 artifact_*.mat，请先完成 200Hz 训练。');

artifact_path = fullfile(candidates(1).folder, candidates(1).name)
S = load(artifact_path);
assert(isfield(S,'net') && isfield(S,'norm'), 'artifact 内缺少 net 或 norm。');

net   = S.net;     % patternnet
normp = S.norm;    % 包含 xmin/xmax

% 2) 生成可代码生成的网络预测函数（features×N -> classes×N）
outDir = fullfile(pwd,'blocks');
if ~exist(outDir,'dir'); mkdir(outDir); end
mfile = fullfile(outDir,'predict_bpn.m');
genFunction(net, mfile, 'MatrixOnly','yes');  % 生成 predict_bpn.m

% 3) 保存归一化参数，供 Simulink 常量块读取
save(fullfile(outDir,'params_norm.mat'),'normp');

fprintf('✅ 导出完成：\n- 网络函数: %s\n- 归一化参数: %s\n', mfile, fullfile(outDir,'params_norm.mat'));
end
