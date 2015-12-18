% Wine type classification using naive bayes classifier

% Spliting training data and validation data 80:20
Wine_train = csvread('training_classification_regression_2015.csv');

Xtrain = double(Wine_train(1:4000,1:end-2));
Ltrain = double(Wine_train(1:4000,end));

Xvalid = double(Wine_train(4001:end,1:end-2));
Lvalid = double(Wine_train(4001:end,end));

%determining the unique classes
Ltrain_unique = unique(Ltrain);

%Number of classes
number_classes=length(Ltrain_unique); 

%Feature set
number_features=size(Xtrain,2);

%Number of samples of Validation set
valid_length=length(Lvalid);

%Determining prior
for i=1:number_classes
    prior_probability(i)=sum(double(Ltrain==Ltrain_unique(i)))/length(Ltrain);
end


for i=1:number_classes
    for k=1:number_features
        parameters=Xtrain(Ltrain==Ltrain_unique(i),k);
        feature_valid=Xvalid(:,k);
        fuStruct(i,k).f=ksdensity(parameters,feature_valid);
    end
end

% re-structure
for i=1:valid_length
    for j=1:number_classes
        for k=1:number_features
            fu(j,k)=fuStruct(j,k).f(i);
        end
    end
    class_probability(i,:)=prior_probability.*prod(fu,2)';
end

% Predict labels
[predicted_labels0,id]=max(class_probability,[],2);
for i=1:length(id)
    predicted_labels(i,1)=Ltrain_unique(id(i));
end

% Compare predicted label with actuall labels
%Confusion Matrix
confMat=myconfusionmat(Lvalid,predicted_labels);
disp('confusion matrix:')
disp(confMat)
% Prediction accuracy
conf=sum(predicted_labels==Lvalid)/length(predicted_labels);
disp(['accuracy = ',num2str(conf*100),'%'])



