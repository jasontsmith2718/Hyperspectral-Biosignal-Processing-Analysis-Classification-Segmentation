function [Sensitivity, Specificity, Accuracy] = loocvclassify(positive,negative)
% [Sensitivity, Specificity, Accuracy] = loocvclassify(positive,negative)
% X = [positive;negative]
X = [positive;negative];

maxScores = [];
for i = 1:size(X,1)
    fprintf('LOOCV_ext: %d\n',i);
    if i <= length(positive)
        positive_train = X([1:i-1 i+1:size(positive,1)],:);
        negative_train = X(size(positive,1)+1:end,:);
    else
        positive_train = X(1:size(positive,1),:);
        negative_train = X([size(positive,1)+1:i-1 i+1:end],:);
    end
    testset = X(i,:);
    trainset = [positive_train;negative_train];

    samples = [repmat({'positive'},size(positive_train,1),1);repmat({'negative'},size(negative_train,1),1)];%repmat({'Intermediate'},length(mid),1)];
    SVMModels = cell(2,1);
%     classes = unique(samples);
    classes = {'positive','negative'};
    rng(1); % For reproducibility

    for j = 1:numel(classes);
        indx = strcmp(samples,classes(j)); % Create binary classes for each classifier
        SVMModels{j} = fitcsvm(trainset,indx,'ClassNames',[false true],'Standardize',true,...
            'KernelFunction','linear','BoxConstraint',1);
    end

    N = size(testset,1);
    Scores = zeros(N,numel(classes));

    for j = 1:numel(classes);
        [~,score] = predict(SVMModels{j},testset);
        Scores(:,j) = score(:,2); % Second column contains positive-class scores
    end
    [~,maxScore] = max(Scores,[],2);
    maxScores(i) = maxScore;
end

positiveScores = maxScores(1:size(positive,1));
negativeScores = maxScores(size(positive)+1:end);
TP = sum(positiveScores==1)
FN = sum(positiveScores==2)
TN = sum(negativeScores==2)
FP = sum(negativeScores==1)

Sensitivity = TP/(TP+FN);
Specificity = TN/(TN+FP);
Accuracy = (TP+TN)/(TP+FN+TN+FP);
fprintf('Sensitivity = %4.1f%%, Specificity = %4.1f%%, Accuracy = %4.1f%%\n',Sensitivity*100,Specificity*100,Accuracy*100);
