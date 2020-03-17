function [Sensitivity,Specificity,Accuracy] = svm2d(negative,positive)

features = [negative;positive];
N_neg = length(negative);
N_pos = length(positive);
N_tot = N_neg + N_pos;

labels = [repmat({'negative'},N_neg,1);repmat({'positive'},N_pos,1)];%repmat({'Intermediate'},length(mid),1)];

SVMModels = cell(2,1);
classes = unique(labels);
rng(1); % For reproducibility
indx = strcmp(labels,'positive'); % Create binary classes for each classifier
for j = 1:numel(classes);
    indx = strcmp(labels,classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(features,indx,'ClassNames',[false true],'Standardize',true,...
    'KernelFunction','linear','BoxConstraint',1000);
end

Scores = zeros(N_tot,numel(classes));
for j = 1:numel(classes);
    [~,score] = predict(SVMModels{j},features);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end
[~,maxScore] = max(Scores,[],2);

TN = sum(maxScore(1:N_neg)==1)          % should be consistent with the index in {classes}
TP = sum(maxScore(N_neg+1:end)==2)      % should be consistent with the index in {classes}

% TP = sum((indx==1).*(Scores>=0));
% FN = sum((indx==1).*(Scores<0));
% TN = sum((indx==0).*(Scores<0));
% FP = sum((indx==0).*(Scores>=0));
Sensitivity = TP./N_pos;
Specificity = TN./N_neg;
Accuracy = (TP+TN)./N_tot;
fprintf('Sensitivity = %4.1f%%, Specificity = %4.1f%%, Accuracy = %4.1f%%\n',Sensitivity*100,Specificity*100,Accuracy*100);

% ROC curve
figure(1)
AUC = svmroc(negative,positive,SVMModels{find(strcmp('positive',classes))},0);
% AUC = svmroc(negative,positive,SVMModel,0);
fprintf('AUROC = %14.7f\n',AUC);

if size(features,2) ~= 2
    return
end
%% classification plot    
% feature1 = features(:,1);
% feature2 = features(:,2);
% d1 = (max(feature1)-min(feature1))/100;
% d2 = (max(feature2)-min(feature2))/100;
% [x1Grid,x2Grid] = meshgrid(min(feature1):d1:max(feature1),min(feature2):d2:max(feature2));
% xGrid = [x1Grid(:),x2Grid(:)];
% N_test = size(xGrid,1);
% Scores = zeros(N_test,numel(classes));
% for j = 1:numel(classes);
%     [~,score] = predict(SVMModels{j},xGrid);
%     Scores(:,j) = score(:,2); % Second column contains positive-class scores
% end
% [~,maxScore] = max(Scores,[],2);

SVMModel = SVMModels{find(strcmp('positive',classes))};
x = features(:,1);
x_std = (x-SVMModel.Mu(1))/SVMModel.Sigma(1);
y_std = -SVMModel.Beta(1)/SVMModel.Beta(2)*x_std-SVMModel.Bias/SVMModel.Beta(2);
% x1 = x1*SVMModel.Sigma(1)+SVMModel.Mu(1);
y = y_std*SVMModel.Sigma(2)+SVMModel.Mu(2);


figure(2);hold on
gscatter(features(:,1),features(:,2),labels,'br','^o');
xlabel('Feature 1');
ylabel('Feature 2');
plot(x,y,'k')
legend('Negative','Positive','Location','Best')


% figure(2);hold on
% h = [];
% h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
% [0.1 0.5 0.5; 0.5 0.1 0.5]);
% hold on
% h(3:4) = gscatter(mu,sigma,grade);
% xlabel('PC1');
% ylabel('PC2');
% legend(h,{'Normal region','Cancer region','Normal','Cancer'});
% title('SVM classification')