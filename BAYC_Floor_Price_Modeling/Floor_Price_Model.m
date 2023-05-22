% BAYC Floor price modeling

% Read the data

bayc = readtable("C:\Users\Roberto\Universidad\TFM\Floor_price\boredapeyachtclub.csv"); % Specify location
date=bayc.date;
y=bayc.price;
plot(date,y)

%Transformation

y_1=y;
for i=2:length(y)-1   
    y_1(i)=median([y(i-1) y(i) y(i+1)]);      % Median filter
end
plot(date,y_1)

y_train=y_1(1:round(length(y_1)*0.7,0));      

[lambda_max, bct] = bcNormPlot(y_train,1);    % Box-cox to stabilize variance
hold on
plot(lambda_max,max(bct),'Marker','o','MarkerSize',6,'Color','r')
plot([lambda_max lambda_max],[max(bct) -4340],'r--')
hold off

% lambda is -0.5, however if that value is used to transform the data, the
% resulting values are very close to each to other, therefore 0.5 is used instead

boxcox_trans = @(y,lambda) (y.^lambda-1)/lambda;
boxcox_trans_inv = @(y,lambda) (y*lambda+1).^(1/lambda);

y_trans=boxcox_trans(y_1,0.5);

y_train_trans = boxcox_trans(y_train,0.5);
plot(date(1:length(y_train)),y_train_trans)

%% Model 1 (AR model)

% Periodicities and trends

y_train_trans=y_train_trans-mean(y_train_trans);
periodogram(y_train_trans,hanning(length(y_train_trans)),16384,1) % No seasonality

% ACF and PACF

subplot(211)
acf(y_train_trans,50,0.05,1,0,0);        % Decaying ACF -> differentiation
title('ACF')
subplot(212)
pacf(y_train_trans,50,0.05,1,0);         % Strong component at lag 1
title('PACF')

A=[1 -1];
y_diff=filter(A,1,y_train_trans);        % Differentiation

% Model estimation

subplot(211)
acf(y_diff,50,0.05,1,0,0);
title('ACF')
subplot(212)
pacf(y_diff,50,0.05,1,0);                % Component at lag 1
title('PACF')

% AR (1)

A=[1 1];
model_init=idpoly(A,[],1);
model_init.Structure.a.Free =[0 1];
ar_model=pem(y_diff,model_init);

rar=resid(ar_model,y_diff);
rar=rar(length(A):end);

subplot(211)
acf(rar.y,50,0.05,1,0,0);
hold on
tacf(rar.y,50,0.02,0.05,1,0);
title('ACF and TACF')
subplot(212)
pacf(rar.y,50,0.05,1,0);
title('PACF')

% Check residuals whiteness

checkIfWhite(rar.y)                       % White
pacfEst = pacf( rar.y, 50, 0.05 );      
checkIfNormal(pacfEst(2:end),'PACF','J')  % But..., not normal PACF

ar_model.a=conv(ar_model.a, [1 -1]);    
 
present(ar_model)                         % Resulting model

% linear predictions

y_validation = y_1(length(y_train)+1:length(y_train)+round(length(y_1)*0.1,0));
y_test= y_1((length(y_validation)+length(y_train)+1):end);

y_validation_trans = boxcox_trans(y_validation, 0.5);
y_test_trans = boxcox_trans(y_test, 0.5);

% Validation data

k=1;
[F,G]=polydiv(ar_model.c,ar_model.a,k);

yhat_kval=filter(G,ar_model.c,y_validation_trans);

yhat_kval = boxcox_trans_inv(yhat_kval,0.5);

res=y_validation-yhat_kval;
res=res(length(G):end);
res_val_var = var(res);

acf(res,50,0.05,1,0,0);   % MA(k-1)

plot(yhat_kval(length(G):end),'r--')
hold on
plot(y_validation(length(G):end),'b')
legend('7-step prediction','True value')
hold off

% Test data

yhat_ktest=filter(G,ar_model.c,y_test_trans);

yhat_ktest = boxcox_trans_inv(yhat_ktest,0.5);

res_AR=y_test-yhat_ktest;
res_AR=res_AR(length(G):end);
res_AR_var = var(res_AR);

plot(yhat_ktest(length(G):end),'r--')
hold on
plot(y_test(length(G):end),'b')
legend('1-step prediction','true value')
hold off

n_train = length(y_train);
n_val = length(y_validation);
n_test = length(y_test);

% Plot with predictions
plot(1:n_train,y_train,'Color',"#000000")
ylim([min(y)-10000 max(y)+10000])
xlim([1 n_val+n_train+n_test])
hold on
plot((n_train):(n_train+n_val),[y_train(end);y_validation],'Color',"#77AC30")
plot((n_train+n_val):(n_train+n_val+length(G)),[y_validation(end); y_test(1:length(G))],'Color',"#77AC30")
plot((n_train+n_val+length(G)):(n_train+n_val+n_test),y_test(length(G):end),'Color',"#0072BD")
line( [n_train+n_val+length(G) n_train+n_val+length(G)],...
    [min(y)-10000 max(y)+10000], 'Color','red','LineStyle',':' )
line( [n_train n_train],...
    [min(y)-10000 max(y)+10000], 'Color','red','LineStyle',':' )
hold on
plot((n_train+n_val+length(G)):length(y),yhat_ktest(length(G):end),'Color',"#D95319")
hold off
legend('Train','Validation','','Test','','','1-step prediction')
%% Model 2 (Kalman Filter)

%Filter initialization

N=length(y);

k=1;
p=2;
q=0;

A=eye(p+q);
Rw=1;
Re=1e-6;
Rx_t1=eye(p+q);
h_et  = zeros(N,1);                             
xt    = [-1.236*ones(1,N);0.2365*ones(1,N)];                         
yhat  = zeros(N,1);                             
yhatk = zeros(N,1); 

for t=3:N-k                                  
    % Update the predicted state and the time-varying state vector
    x_t1 = A*xt(:,t-1);                         
    C    = [-y_trans(t-1) -y_trans(t-2)];     
    
    % Update the parameter estimates
    Ry = C*Rx_t1*C' + Rw;                    
    Kt = Rx_t1*C'/Ry;                        
    yhat(t) = C*x_t1;                          
    h_et(t) = y_trans(t)-yhat(t);                     
    xt(:,t) = x_t1 + Kt*( h_et(t) );            

    % Update the covariance matrix estimates
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                 
    Rx_t1 = A*Rx_t*A' + Re;                   

    % Form the k-step prediction 
    Ck = [-y_trans(t) -y_trans(t-1)];           
    yk = Ck*xt(:,t);                         

    yhatk(t+k) = boxcox_trans_inv(yk,0.5);                              
end

% Plot
plot(yhatk(15:end),color="#D95319")
hold on 
plot(y_1(15:end),color="#0072BD")
legend('Kalman Filter','Original data')
hold off

% Parameters comparison
xt=xt';
plot(xt((n_train+n_val):end-k,1),color="#D95319")
hold on
plot(xt((n_train+n_val):end-k,2),color='blue')
a1=-1.236*ones(size(xt((n_train+n_val):end-k,:)));
a2=0.2365*ones(size(xt((n_train+n_val):end-k,:)));
plot(a1,color="#D95319",LineStyle='--')
plot(a2,color='blue',LineStyle='--')
legend('AR(1) Kalman Filter','AR(2) Kalman Filter','AR(1) Model 1','','AR(2) Model 1','Location','east')
hold off

% Variance comparison
res_AR_var;
res_kalman=y_test-yhatk((n_train+n_val+1):end);
res_kalman_var=var(res_kalman);                   % Higher variance for the kalman filter

checkIfWhite(res_AR)                              % White
checkIfWhite(res_kalman)                          % Not white

acf(res_kalman,50,0.05,1,0,0);
acf(res_AR,50,0.05,1,0,0);