% windreader.m subscript for missing data treatment
idxs = find(isnan(X(:,end)));
for n=1:length(idxs)
    idx_alt = find(nwp(:,1)==nwp(use_idx(idxs(n)),1)); % look for alternative older weather predictions
    idx_alt = idx_alt(~isnan(nwp(idx_alt,missdatcol)));
    idx_alt = idx_alt(nwp(idx_alt,3)<nwp(use_idx(idxs(n)),3)); % must be older than current forecast
    if ~isempty(idx_alt) 
        idx_alt = idx_alt(end); %most recent altenative forecast
        X(idxs(n),end) = nwp(idx_alt,missdatcol); 
        [dumY,dumMO,dumD,dumH] = datevec(nwp1TS((idx_alt))-nwp3TS((idx_alt))); %new age
        H(idxs(n)) = dumH + dumD*24;
    else
        H(idxs(n)) = NaN;
    end %else leave X-NaN untouched
end

  