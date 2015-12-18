% Wine type classification using naive bayes classifier

% Reading training data 
Wine_train = csvread('training_classification_regression_2015.csv');
Wine_test= csvread('challenge_public_test_classification_regression_2015.csv');

Xtrain = double(Wine_train(:,1:end-2));
Ltrain = double(Wine_train(:,end));

Xtest = double(Wine_test(:,2:end-2));

%determining the unique classes
Ltrain_unique = unique(Ltrain);

%Number of classes
number_classes=length(Ltest_unique); 

%Feature set
number_features=size(Xtrain,2);

%Number of samples of Validation set
valid_length=length(Ltrain);

%Determining prior
for i=1:number_classes
    prior_probability(i)=sum(double(Ltrain==Ltrain_unique(i)))/length(Ltrain);
end


for i=1:number_classes
    for k=1:number_features
        parameters=Xtest(Ltrain==Ltrain_unique(i),k);
        feature_valid=Xtest(:,k);
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
    predicted_labels(i,1)=Ltest_unique(id(i));
end

% Write predicted labels to output files
wine_ids=Wine_test(:,1);
final = [wine_ids predicted_labels];
csvwrite('output_type_naives.csv',final,'');




