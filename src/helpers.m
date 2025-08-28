classdef helpers
methods(Static)
    function [fpr, tpr, thr, auc_macro, h] = plot_roc_ova(Y_true_onehot, Yhat_prob)
        % One-vs-All ROC，返回 macro AUC
        classes = size(Y_true_onehot, 2);
        aucs = zeros(classes,1);
        h = figure('Name','ROC (OvA)'); hold on; grid on;
        for c = 1:classes
            [X,Y,~,AUC] = perfcurve(Y_true_onehot(:,c), Yhat_prob(:,c), 1);
            plot(X,Y,'LineWidth',1.5);
            aucs(c) = AUC;
        end
        auc_macro = mean(aucs);
        xlabel('False Positive Rate'); ylabel('True Positive Rate');
        title(sprintf('ROC OvA (macro AUC=%.3f)', auc_macro));
        legend(arrayfun(@(k)sprintf('Class %d',k-1), 1:classes, 'UniformOutput', false), 'Location','SouthEast');
        fpr = []; tpr = []; thr = [];
    end
end
end
