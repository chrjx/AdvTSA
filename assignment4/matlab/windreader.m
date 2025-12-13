function [Y,X,X_legend,nwp,pow] = windreader(pow_lags,WFC_DTs)
% Import of wind data for exercise 4a in 02427 Adv. Time Series Analysis
%
% [Y,X,X_legend,nwp,pow] = windreader(pow_lags,WFC_DTs)
%
% Y: The response variable to be predicted (i.e. pow_t+1 pow_t+2 pow_t+3).
% X: The designmatrix containing your basic features/regressors.
% X_legend: Contains a discription of the included features.
% nwp, and pow: Raw output of the available data.
%
% pow_lags: Specify the included lags [hours] of power observations in X 
% WFC_DTs:  Specify the included weather forecasts [hours] in X 
%
% The included weather forecast is the best available in data for a given point
% in time.  The "length" of a weather forecast is the time difference between 
% time of prediction and time of forecast calculation (current time may be in 
% between or after). "length" may increase due to missing forecasts. When there 
% is no more alternative forecasts, the forecast is missing which is indicated 
% by NaN's.
%
% pow_lags = [0 1 3]; % arbitrary example
% WFC_DTs  = [-48 -24 0 1 2 3 24];  % arbitrary example
%
% By: Kristian Tjalfe Klinkby (ktkl@imm.dtu.dk) - dec.2010


if max(WFC_DTs) > 43
    error('Maximal 43 hours weather forecast are allowed.')
    return
end

if min(pow_lags) < 0
    error('Power lags must be non-negative.')
    return
end


fid = fopen('/xbar/nas2/home2/imm/hmad/02427/wind_nwp.dep');
for n=1:16
    fgetl(fid);
end
nwp= fscanf(fid,'%f %f %f %f %f %f',[6 inf])';
fclose(fid);

nwp(nwp(:,6)==0,5:6) = NaN; % odd direction jump and 0 kelvin? Not likely!

pow=dlmread('/xbar/nas2/home2/imm/hmad/02427/wind_pow.dep','\t',12,0);

snwpdate1 =  num2str(nwp(:,1));
snwpdate3 =  num2str(nwp(:,3));
spowdate  =  num2str(pow(:,1));

%time of day:
%nwptod1 = base2dec(snwpdate1(:,9:10),10);
%nwptod3 = base2dec(snwpdate3(:,9:10),10);
powtod  = base2dec(spowdate(:,9:10), 10);

%timestamp:
nwp1TS = datenum(snwpdate1,'yyyymmddHHMM');
nwp3TS = datenum(snwpdate3,'yyyymmddHHMM');
%powTS  = datenum(spowdate, 'yyyymmddHHMM');


NYX = length(pow)-3;
NW  = length(nwp);
idx_base = kron(0:49:(NW-49),ones(1,6)) + kron(ones(1,NW/49),1:6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%target for prediction: 1,2 and 3 hours ahead:
Y = [pow(2:end-2,3) pow(3:end-1,3) pow(4:end,3)];

timestamp = pow(1:end-3,1); % timepoint for our generated prediction

%%%%% time features
X = timestamp;
clear X_legend;
X_legend{1} = 'Timestamp';

X(:,end+1) = pow(1:end-3,2);
X_legend{2,1} = 'Time of year';

X(:,end+1) = powtod(1:end-3);
X_legend{end+1} = 'Time of day';


%%%%% lag-delays for power-feature:
idx0 = size(X,2);
for idx=1:length(pow_lags)
    X(:,idx+idx0) = filter([zeros(1,pow_lags(idx)) 1],1,pow(1:end-3,3));
    X(1:pow_lags(idx),idx+idx0) = NaN; %missing data
    X_legend{idx+idx0} = ['Power lag ' num2str(pow_lags(idx))];
end

use_idx = idx_base(1:end-3); 
leg_atoms{1} = ' wind speed';
leg_atoms{2} = ' wind direction';
leg_atoms{3} = ' temperature';
%%%%%  Current weather forecast

for k=1:3 % wind speed, wind direction, temperature
    datcol = k+3;  
    CurrWFC(:,k) = nwp(use_idx,datcol);
    [~,~,D,H] = datevec(nwp1TS(use_idx)-nwp3TS(use_idx)); %prognosis length
    [Xdat,H] = missdat(CurrWFC(:,k),H+24*D,nwp,use_idx,datcol,nwp1TS,nwp3TS);
    CurrWFC(:,k) = Xdat;
    CurrWFClength(:,k) = H;  % updated length
end

%%%%% Weather forecast features
for k=1:3 % wind speed, wind direction, temperature
    for j=1:length(WFC_DTs)
        
        if WFC_DTs(j)<0
            X(:,end+1) = filter([zeros(1,-WFC_DTs(j)) 1],1,CurrWFC(:,k),zeros(1,-WFC_DTs(j))+NaN);
            X_legend{end+1} = ['Forecasted t' num2str(WFC_DTs(j)) leg_atoms{k}];
            X(:,end+1) = filter([zeros(1,-WFC_DTs(j)) 1],1,CurrWFClength(:,k),zeros(1,-WFC_DTs(j))+NaN);
            X_legend{end+1} = ['Length of t' num2str(WFC_DTs(j)) leg_atoms{k} ' forecast'];
        elseif WFC_DTs(j)==0
            X(:,end+1) = CurrWFC(:,k);
            X_legend{end+1} = ['Forecasted t+0' leg_atoms{k}];
            X(:,end+1) = CurrWFClength(:,k);
            X_legend{end+1} = ['Length of t+0' leg_atoms{k} ' forecast'];
        else
            datcol = k+3;
            use_idx = idx_base(1:end-3)+WFC_DTs(j);
            Xdat = nwp(use_idx,datcol) ;
            
            [~,~,D,H] = datevec(nwp1TS(use_idx)-nwp3TS(use_idx)); %prognosis length
            [Xdat,H] = missdat(Xdat,H+24*D,nwp,use_idx,datcol,nwp1TS,nwp3TS);
            X(:,end+1) = Xdat;  % updated data
            X_legend{end+1} = ['Forecasted t+' num2str(WFC_DTs(j)) leg_atoms{k}];
            
            X(:,end+1) = H;  % updated prognosis length
            X_legend{end+1} = ['Length of t+' num2str(WFC_DTs(j)) leg_atoms{k} ' forecast'];
        end
    end
end



function  [X,H] = missdat(X,H,nwp,use_idx,missdatcol,nwp1TS,nwp3TS)
% windreader.m subscript for missing data treatment
idxs = find(isnan(X(:,end)));
for n=1:length(idxs)
    idx_alt = find(nwp(:,1)==nwp(use_idx(idxs(n)),1)); % look for alternative older weather predictions
    idx_alt = idx_alt(~isnan(nwp(idx_alt,missdatcol)));
    idx_alt = idx_alt(nwp(idx_alt,3)<nwp(use_idx(idxs(n)),3)); % must be older than current forecast
    if ~isempty(idx_alt) 
        idx_alt = idx_alt(end); %most recent altenative forecast
        X(idxs(n),end) = nwp(idx_alt,missdatcol); 
        [~,~,dumD,dumH] = datevec(nwp1TS((idx_alt))-nwp3TS((idx_alt))); %new age
        H(idxs(n)) = dumH + dumD*24;
    else
        H(idxs(n)) = NaN;
    end %else leave X-NaN untouched
end

  
