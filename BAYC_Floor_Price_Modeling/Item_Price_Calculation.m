% Price estimaton based on floor price forecast and rarity score

rarity_score_csv = readtable("C:\Users\Roberto\Universidad\TFM\Floor_price\rarity_score.csv"); % Specify location
rarity_scores = rarity_score_csv.rarity_score;

normalizedScore = @(rarity_score)(rarity_score - min(rarity_scores))/... 
    (max(rarity_scores) - min(rarity_scores));   % Normalization

k=5;              % This constant controls the rate increase in the exponential interpolation

token_price = @(token_rarity,floor_price)floor_price*(1+2/(1-exp(-k))*...
    (exp(-k*normalizedScore(token_rarity))-exp(-k)));  %Exponential interpolation

% Example: Prediction of the price of a random token the date after the
% data was gathered (9th of March)

bayc = readtable("C:\Users\Roberto\Universidad\TFM\Floor_price\boredapeyachtclub.csv"); % Specify location
y=bayc.price;

boxcox_trans = @(y,lambda) (y.^lambda-1)/lambda;
boxcox_trans_inv = @(y,lambda) (y*lambda+1).^(1/lambda);

ar_model=[1.236 -0.2365];

y_trans=boxcox_trans(y,0.5);

floor_price = filter(ar_model,1,y_trans(end-1:end));
floor_price = floor_price(length(ar_model):end);
floor_price = boxcox_trans_inv(floor_price,0.5);

token_price(rarity_scores(10),floor_price)     % Value at index 10 is picked randomly
