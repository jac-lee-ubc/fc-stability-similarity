% ===================================================================================
%  Example LME: 
%  Relationship between Age and Cross-State Stability (WB & SB) 
%
%  Model:
%   - Stability ~ Age * BrainType + MeanSim + Sex + RMSD
%              + (Age| Subject) + (1 | Site)
%
%  Requirements:
%   - 'all_results' is a {subj x 1} cell array that contains subject data:
%       all_results{s}.out.<fieldname>
% ===================================================================================

%% -------------------------------
% 1) Extract and Z-score variables
% -------------------------------
S = numel(all_results);

% Age (IV)
Age = arrayfun(@(s) s.out.MRI_TrackAge_at_Scan, [all_results{:}])';
Age_z = (Age - mean(Age,'omitnan')) ./ std(Age,'omitnan');

% WB & SB stability (DVs)
WB_stab = arrayfun(@(s) s.out.stability_Movie_Rest, [all_results{:}])';
WB_stab_z = (WB_stab - mean(WB_stab,'omitnan')) ./ std(WB_stab,'omitnan');
SB_stab = arrayfun(@(s) s.out.SB_stability_Movie_Rest, [all_results{:}])';
SB_stab_z = (SB_stab - mean(SB_stab,'omitnan')) ./ std(SB_stab,'omitnan');

% Motion
Motion = arrayfun(@(s) s.out.mean_rmsd_full, [all_results{:}])';
Motion_z = (Motion - mean(Motion,'omitnan')) ./ std(Motion,'omitnan');

% Sex
Sex = arrayfun(@(s) s.out.Basic_DemosSex, [all_results{:}])';
Sex = categorical(Sex, [0 1], {'Male','Female'});
Sex = reordercats(Sex, {'Male','Female'});

% Site
Site = arrayfun(@(s) s.out.Basic_DemosStudy_Site, [all_results{:}])';

% WB and SB mean similarity covariate
WB_meanSim = arrayfun(@(s) s.out.mean_similarity_to_healthy, [all_results{:}])';
WB_meanSim_z = (WB_meanSim - mean(WB_meanSim,'omitnan')) ./ std(WB_meanSim,'omitnan');
SB_meanSim = arrayfun(@(s) s.out.SB_mean_similarity_to_healthy, [all_results{:}])';
SB_meanSim_z = (SB_meanSim - mean(SB_meanSim,'omitnan')) ./ std(SB_meanSim,'omitnan');

%% -------------------------------
% 2) Build long table 
% -------------------------------
nSubj = S;
idx = (1:S)';

T = table;
T.Subject = categorical([idx; idx]);
T.Brain   = categorical([zeros(nSubj,1); ones(nSubj,1)], [0 1], {'WB','SB'});
T.FC      = [WB_stab_z; SB_stab_z];
T.Age     = [Age_z; Age_z];
T.MeanSim = [WB_meanSim_z; SB_meanSim_z];
T.Sex     = [Sex; Sex];
T.Motion  = [Motion_z; Motion_z];
T.Site    = categorical([Site; Site]);

T = rmmissing(T);
T.Brain = reordercats(T.Brain, {'WB','SB'});

%% -------------------------------
% 3) Fit LME 
% -------------------------------
lme = fitlme(T, ...
    'FC ~ Age*Brain + MeanSim + Sex + Motion + (1|Site) + (Age|Subject)', ...
    'FitMethod', 'REML');
