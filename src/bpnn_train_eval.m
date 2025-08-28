function R = bpnn_train_eval(dsTrain, dsVal, dsTest, normp, C, tag)
% 训练 + 评估 + 绘图导出（修正了样本数断言）
if ~exist('../figures','dir'); mkdir('../figures'); end
if ~exist('../results','dir'); mkdir('../results'); end
if isfield(C,'seed'), rng(C.seed,'twister'); end

% === 基础尺寸检查（行=样本） ===
assert(size(dsTrain.X,1) == size(dsTrain.Y,1), 'Train X/Y 样本数不一致');
assert(size(dsVal.X,1)   == size(dsVal.Y,1),   'Val   X/Y 样本数不一致');
assert(size(dsTest.X,1)  == size(dsTest.Y,1),  'Test  X/Y 样本数不一致');

% === 构建网络 ===
net = patternnet(C.hidden_sizes, C.trainFcn);
net.performFcn = C.performFcn;

% 使用 divideind 手动指定划分（patternnet 以“列”为样本”）
net.divideFcn = 'divideind';

% 统一转置为 [特征×样本] / [类别×样本]
Xtr = dsTrain.X'; Ytr = dsTrain.Y';
Xva = dsVal.X';   Yva = dsVal.Y';
Xte = dsTest.X';  Yte = dsTest.Y';

% 按列拼接（列=样本）
Xall = [Xtr, Xva, Xte];
Yall = [Ytr, Yva, Yte];

Ntr = size(Xtr,2);
Nva = size(Xva,2);
Nte = size(Xte,2);

idxTr = 1:Ntr;
idxVa = Ntr + (1:Nva);
idxTe = Ntr + Nva + (1:Nte);

net.divideParam.trainInd = idxTr;
net.divideParam.valInd   = idxVa;
net.divideParam.testInd  = idxTe;

net.trainParam.epochs          = C.max_epoch;
net.trainParam.showWindow      = C.showWindow;
net.trainParam.max_fail        = C.max_fail;


if isfield(C,'showCommandLine')
    net.trainParam.showCommandLine = C.showCommandLine;
else
    net.trainParam.showCommandLine = true;
end

% === 训练 ===
[net, tr] = train(net, Xall, Yall);

% === 预测 ===
Yhat_all = net(Xall);              % [K x (Ntr+Nva+Nte)]
Yhat_tr  = Yhat_all(:, idxTr);
Yhat_va  = Yhat_all(:, idxVa);
Yhat_te  = Yhat_all(:, idxTe);

% 交叉熵（test）
loss_xent = perform(net, Yte, Yhat_te);

% === 指标：混淆矩阵、精确率/召回率、AUC/ROC ===
% compute_metrics 需要 [N x K]，因此转置
[acc, prec, rec, cm] = compute_metrics(dsTest.Y, Yhat_te');

% === 绘图并保存 ===
h1 = figure('Name',['Performance_' tag]);
if ~isempty(tr) && isfield(tr,'perf')
    plotperform(tr);
else
    title('Training performance (record unavailable)');
end
saveas(h1, fullfile('../figures', sprintf('perform_%s.png', tag)));

h2 = figure('Name',['Confusion_' tag]);
plotconfusion(dsTest.Y', Yhat_te);     % 目标/预测均为 [K×N]
saveas(h2, fullfile('../figures', sprintf('confusion_%s.png', tag)));

[~,~,~,auc_macro, h3] = helpers.plot_roc_ova(dsTest.Y, Yhat_te'); % [N×K]
saveas(h3, fullfile('../figures', sprintf('roc_%s.png', tag)));

% === 导出结果 ===
R.acc_overall   = acc;
R.prec_macro    = mean(prec);
R.recall_macro  = mean(rec);
R.auc_macro     = auc_macro;
R.loss_xent     = loss_xent;
R.cm            = cm;
R.tag           = tag;
R.Yhat_test     = Yhat_te';     % [N x K]
R.model         = net;

S = struct('config', C, 'norm', normp, 'results', R, ...
           'train', dsTrain, 'val', dsVal, 'test', dsTest, 'net', net);
save(fullfile('../results', sprintf('artifact_%s.mat', tag)), '-struct', 'S');

fprintf('\n[%s] Accuracy=%.3f  CE-loss=%.4f  Prec(m)=%.3f  Rec(m)=%.3f  AUC(m)=%.3f\n', ...
    tag, R.acc_overall, R.loss_xent, R.prec_macro, R.recall_macro, R.auc_macro);
end
