function pan= panMP(TS)

TS = TS(:);        % Make sure the vector is in column format 

MinMotifLen = 32;
MaxMotifLen = 512;
StepSize    = 64;
MotifPlotMode = true;  % If true, the plot is flipped upside down, so that motifs are peaks, not valleys 

for i = MinMotifLen : StepSize : MaxMotifLen

   disp(['Motif length ',num2str(i), ' of ',num2str(MaxMotifLen),]);            % Echo meta data
   [matrixProfile, matrixProfileIdx] = mpxForPan(TS, i-2, i);                   % Compute the MP
   LengthNormalizedMP = zscore((matrixProfile./(sqrt(i)))') ;                   % Correct for length (the 'sqrt(i)' step) and normalize
   LengthNormalizedMP = LengthNormalizedMP(ones(1, StepSize), :)';              % If the StepSize is greater than 1, duplicate the MP StepSize times 
   pan(1:length(matrixProfile),i:i+StepSize-1) = LengthNormalizedMP;            % Insert the computed MP into the Pan Surface plot
   
end


if MotifPlotMode                % User wants a motif plot, so flip it upside down
    pan = pan - max(max(pan));  % Normalize surface to peak at one.  
    pan = pan *-1;
end

pan(end-MaxMotifLen:end,:) = [];  % Trim spurious data from end of surface plot
pan = flipud(pan);
pan(:,1:MinMotifLen-1) = nan;     % Set data from 1 to MinMotifLen to NaN. which is invisible in Matlab

pan= pan./(max(max(pan)));        % Normalize, and plot
figure; s= surf((pan));  s.EdgeColor = 'none';
s.FaceColor = 'interp';
colormap jet(1000)

end