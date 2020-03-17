function AUC = svmroc(negative,positive,SVMModel,mode)
% f(x) = w'x + b
% AUC = svmroc(negative,positive,SVMModel,mode)
% positive: N_posxM marix, where N_pos is the number of postive samples, M is the number of features.
% negative: N_negxM matrix, where N_neg is the number of negative samples.
% SVMModel: SVM Model created by fitcsvm, created with "positive" as true in second column, samples as [negative;positive]]
% mode: 0 - built-in function, fitPosterior and perfcurve
%       1 - fix w, vary b

X = [negative;positive];
samples = [repmat({'negative'},size(negative,1),1);repmat({'positive'},size(positive,1),1)];
samples_label = strcmp(samples,'positive');
% SVMModel = fitcsvm(X,samples_label,'ClassNames',[false true],'Standardize',true,'KernelFunction','linear','BoxConstraint',10);

if mode ~= 0
    % ---------- ROC - vary b, keep w the same, f(x) = w'x + b --------
    % For 2d:
    % x1,y1: standardized; x2,y2: raw; x = [x1,y1]'.
    % Hyperplane function is: x'*Beta + Bias = 0, i.e. [x1,y1]*[Beta(1);Beta(2)] + Bias = 0.
    % So, x1*Beta(1)+y1*Beta(2) + Bias = 0.
    % x_std = (X(:,1)-SVMModel.Mu(1))/SVMModel.Sigma(1);
    % y_std = -SVMModel.Beta(1)/SVMModel.Beta(2)*x_std-SVMModel.Bias/SVMModel.Beta(2);
    % x = x_std*SVMModel.Sigma(1)+SVMModel.Mu(1);
    % y = y_std*SVMModel.Sigma(2)+SVMModel.Mu(2);
    % distance = (X(:,2)-y2)/norm(SVMModel.Beta)
    if ~isempty(SVMModel.Mu)
        scoreROC = (X-repmat(SVMModel.Mu,size(X,1),1))./repmat(SVMModel.Sigma,size(X,1),1)*SVMModel.Beta+SVMModel.Bias;
    else
        scoreROC = X*SVMModel.Beta+SVMModel.Bias;
    end
    b_START = SVMModel.Bias - max(scoreROC) - 1;
    b_END = SVMModel.Bias - min(scoreROC) + 1;
    stepsize = (b_END - b_START)/1000;
    Sensitivity = [];Specificity = [];Accuracy = [];
    i = 1;
    for b = b_START:stepsize:b_END
        if ~isempty(SVMModel.Mu)
            scoreROC = (X-repmat(SVMModel.Mu,size(X,1),1))./repmat(SVMModel.Sigma,size(X,1),1)*SVMModel.Beta + b;
        else
            scoreROC = X*SVMModel.Beta + b;
        end
        TP = sum((samples_label==1).*(scoreROC>=0));
        FN = sum((samples_label==1).*(scoreROC<0));
        TN = sum((samples_label==0).*(scoreROC<0));
        FP = sum((samples_label==0).*(scoreROC>=0));
        Sensitivity(i) = TP./(TP+FN);
        Specificity(i) = TN./(TN+FP);
        Accuracy(i) = (TP+TN)./(TP+FN+TN+FP);
        i = i+1;
    %     fprintf('Sensitivity = %f, Specificity = %f, Accuracy = %f\n',Sensitivity,Specificity,Accuracy);
    end
    plot((1-Specificity),Sensitivity,'k','LineWidth',2);
    set(gca,'FontSize',12);
    xlabel('1-Specificity')
    ylabel('Sensitivity')

    % AUC = trapz(1-Specificity,Sensitivity)
    xy = [1-Specificity' Sensitivity'];
    xy = sortrows(xy,1);
    AUC = trapz(xy(:,1),xy(:,2))

else
    % -------------- ROC - use built-in fitPosterior using sigmoid function ----------------
    SVMModel = fitPosterior(SVMModel); %'Leaveout','on'
    [~,score_svm] = resubPredict(SVMModel);
    [Xsvm,Ysvm,Tsvm,AUC] = perfcurve(samples_label,score_svm(:,SVMModel.ClassNames),'true');
    if AUC < 0.5
        AUC = 1-AUC;
        Xsvm = 1-Xsvm;
        Ysvm = 1-Ysvm;
        plot(Xsvm,Ysvm,'k-','LineWidth',2);
    else
        plot(Xsvm,Ysvm,'k-','LineWidth',2);
    end
    set(gca,'FontSize',12);
    xlabel('1-Specificity')
    ylabel('Sensitivity')
end