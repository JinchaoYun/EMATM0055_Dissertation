function [train, val, test, normp] = split_normalize(X, Y, C)
% 分层随机划分 + 训练集全局 min–max 归一化（保持输出与现有代码兼容）
% 输入:
%   X: [N x D]  行=样本, 列=特征
%   Y: [N x K]  行=样本, 列=one-hot 类别
% 输出:
%   train/val/test: .X 为 [N x D], .Y 为 [N x K]（保持原状）
%   normp: 训练集全局 min/max

    % ===== 基本参数 =====
    if isfield(C, 'seed'), rng(C.seed, 'twister'); end
    N = size(X,1);
    K = size(Y,2);               % 类别数（建议用Y推断而不是硬编码7）
    assert(mod(N, K)==0, '样本数必须能被类别数整除');
    Nper = N / K;

    rat = C.train_val_test;      % 例如 [0.70 0.15 0.15]

    train_X = []; train_Y = [];
    val_X   = []; val_Y   = [];
    test_X  = []; test_Y  = [];

    % ===== 按类打乱后再切分 =====
    for c = 1:K
        s = (c-1)*Nper + 1; e = c*Nper;
        idx_c = (s:e).';
        rp = randperm(Nper);         % 关键：类内打乱
        idx_c = idx_c(rp);

        ntr = round(rat(1)*Nper);
        nva = round(rat(2)*Nper);
        nts = Nper - ntr - nva;

        idx_tr = idx_c(1:ntr);
        idx_va = idx_c(ntr+1:ntr+nva);
        idx_te = idx_c(ntr+nva+1:end);

        train_X = [train_X; X(idx_tr,:)];   train_Y = [train_Y; Y(idx_tr,:)];
        val_X   = [val_X;   X(idx_va,:)];   val_Y   = [val_Y;   Y(idx_va,:)];
        test_X  = [test_X;  X(idx_te,:)];   test_Y  = [test_Y;  Y(idx_te,:)];
    end

    % ===== 仅用训练集统计 min/max 并归一化 =====
    xmin = min(train_X, [], 1);
    xmax = max(train_X, [], 1);
    eps_ = 1e-9;

    normp.xmin = xmin;
    normp.xmax = xmax;
    normp.eps  = eps_;

    scale = @(A) (A - xmin) ./ max(xmax - xmin, eps_);
    clip01 = @(A) min(max(A, 0), 1);

    train.X = clip01(scale(train_X));   train.Y = train_Y;
    val.X   = clip01(scale(val_X));     val.Y   = val_Y;
    test.X  = clip01(scale(test_X));    test.Y  = test_Y;
end
