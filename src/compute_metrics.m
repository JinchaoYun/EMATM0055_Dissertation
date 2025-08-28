function [acc, precision_per_class, recall_per_class, cm] = compute_metrics(Y_true_onehot, Yhat_prob)
% Y_true_onehot: N×7
% Yhat_prob:     N×7 (概率或得分)
[~, ytrue] = max(Y_true_onehot, [], 2);
[~, ypred] = max(Yhat_prob, [], 2);

cm = confusionmat(ytrue, ypred, 'Order', 1:7);
acc = sum(diag(cm)) / sum(cm(:));

precision_per_class = zeros(7,1);
recall_per_class    = zeros(7,1);
for c = 1:7
    TP = cm(c,c);
    FP = sum(cm(:,c)) - TP;
    FN = sum(cm(c,:)) - TP;
    precision_per_class(c) = TP / max(TP+FP, eps);
    recall_per_class(c)    = TP / max(TP+FN, eps);
end
end
