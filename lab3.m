%% information
% facial age estimation
% regression method: linear regression

%% settings
clear;
clc;

% path 
database_path = './data_age.mat';
result_path = './results/';

% initial states
absTestErr = 0;
cs_number = 0;


% cumulative error level
err_level = 5;

%% Training 
load(database_path);

nTrain = length(trData.label); % number of training samples
nTest  = length(teData.label); % number of testing samples
xtrain = trData.feat; % feature
ytrain = trData.label; % labels

w_lr = regress(ytrain,xtrain);
   
%% Testing
xtest = teData.feat; % feature
ytest = teData.label; % labels

yhat_test = xtest * w_lr;

%% Compute the MAE and CS value (with cumulative error level of 5) for linear regression
%%%MAE
final_error = yhat_test - ytest;
final_error = abs(final_error);
MAE_Lin_Reg = mean(final_error)
disp('Mean Absolute Error:')
disp(MAE_Lin_Reg)

%%%CS
correct = abs(yhat_test - ytest) <= err_level;
correct_length =length(find(correct));
cs_number_Lin_Reg = (correct_length ./ length(ytest));
disp('Cumulative Score:')
disp(cs_number_Lin_Reg)

%% generate a cumulative score (CS) vs. error level plot by varying the error level from 1 to 15.
% error_level_list = 1:15; 
% cs_list = []; 

for r = 1:15
    correct = abs(yhat_test - ytest) <= r;
%     correct_length =length(find(correct));
    CS(r) = (sum(correct) ./ length(ytest));
end

plot(CS, '-bo');
xlabel('Error level');
ylabel('Cumulative score');
title('Cumulative score vs Error level');

%% Compute the MAE and CS value (with cumulative error level of 5) for both partial least square regression and the regression tree model by using the Matlab built in functions.
%%% Partial Least Square Regression model (PLS)
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(xtrain,ytrain,err_level);
yfit_PLS = [ones(size(xtest,1),1) xtest] * beta;

%%% Mean Absolute Error(MAE)
final_error = yfit_PLS - ytest;
final_error = abs(final_error);
MAE_PartialLeastSqr = mean(final_error);
disp('MAE Partial Least Square:');
disp(MAE_PartialLeastSqr);

%%% Cumulative Score(CS)
y_correct = abs(yfit_PLS - ytest) <= err_level;
y_correct_lenght = length(find(y_correct));
CS_PartialLeastSqr = (y_correct_lenght ./ length(ytest));
disp('CS Partial Least Square:');
disp(CS_PartialLeastSqr);

%%%regression tree model
model = fitrtree(xtrain, ytrain);
yfit_RegTree = predict(model, xtest);

%%MAE
error = yfit_RegTree - ytest;
error = abs(error);
MAE_RegTree = mean(error);
disp('MAE Regression Tree:');
disp(MAE_RegTree);

%%CS
y_correct = abs(yfit_RegTree - ytest) <= err_level;
y_correct_length = length(find(y_correct));
CS_RegTree = (y_correct_length ./ length(ytest));
disp('CS Regression Tree:');
disp(CS_RegTree);



%% Compute the MAE and CS value (with cumulative error level of 5) for Support Vector Regressionby using LIBSVM toolbox
cmd = ['-s 3 -t 0 -b 1'];
model = svmtrain(ytrain, xtrain, cmd);
    
    
%%toc 
%%options=sprintf('-s 0 -t 2 -c %f -b 1 -g %f -q',bestc,bestg);
%%model=svmtrain(ytrain, nTrain,options);
%% %% apply the SVM model to the test images 
%{%
[predict_label, accuracy , dec_values] = svmpredict(ytest,xtest, model,'-b 1');
%}
MAE = sum(abs(ytest-predict_label)) ./ length(ytest);
disp('MAE SVR:');
disp(MAE);
CS = sum(abs(ytest-predict_label) <= 5) ./ length(ytest);
disp('CS SVR:');
disp(CS);