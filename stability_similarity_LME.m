% ============================================================
%  Relationship between Cross-State Stability and Within-State Similarity
%  (Whole-Brain vs Social-Brain connectomes)
%
%  Model:
%   - Stability ~ Similarity * BrainType + Age + Sex + RMSD
%              + (Similarity | Subject) + (1 | Site)
%
%  Requirements:
%   - 'all_results' is a {subj x 1} cell array that contains subject data:
%       all_results{s}.out.<fieldname>
%   - MATLAB (fitlme, coefTest)
% ============================================================

%% -------------------------------
% 1) CONFIG: Field names
% -------------------------------

% Whole-brain (WB) measures stored in all_results{s}.out
wb_fields = struct( ...
   'stab',    'stability_Movie_Rest', ...        
   'meanSim', 'mean_similarity_to_healthy');     

% Social-brain (SB) measures stored in all_results{s}.out
sb_fields = struct( ...
   'stab',    'SB_stability_Movie_Rest', ...     
   'meanSim', 'SB_mean_similarity_to_healthy');  

% Covariates (continuous RMSD; binary sex; continuous age)
covariates = {'mean_rmsd_full','Basic_DemosSex','MRI_TrackAge_at_Scan'};

% Site field (categorical)
site_field = 'Basic_DemosStudy_Site';

% Number of subjects
S = numel(all_results);
assert(S > 0, 'all_results is empty.');

%% -------------------------------
% 2) Extract per-subject variables
% -------------------------------

WB_stab = arrayfun(@(s) s.out.(wb_fields.stab),     [all_results{:}])';
SB_stab = arrayfun(@(s) s.out.(sb_fields.stab),     [all_results{:}])';
WB_sim  = arrayfun(@(s) s.out.(wb_fields.meanSim),  [all_results{:}])';
SB_sim  = arrayfun(@(s) s.out.(sb_fields.meanSim),  [all_results{:}])';

RMSD     = arrayfun(@(s) s.out.(covariates{1}), [all_results{:}])';
Sex_raw  = arrayfun(@(s) s.out.(covariates{2}), [all_results{:}])';
Age      = arrayfun(@(s) s.out.(covariates{3}), [all_results{:}])';
Site_raw = arrayfun(@(s) s.out.(site_field),    [all_results{:}])';

% Subject IDs (string)
SubID = arrayfun(@(s) s.out.subject, [all_results{:}], 'UniformOutput', false)';
SubID = string(SubID);

%% -------------------------------
% 3) Stack into long format (2 rows per subject: WB, SB)
% -------------------------------

N = 2*S;

Subject   = categorical(repelem(SubID, 2, 1));
Site      = categorical(repelem(Site_raw, 2, 1));
BrainType = categorical(repmat(["WB"; "SB"], S, 1));

Stability  = NaN(N,1);
Similarity = NaN(N,1);

Stability(BrainType=="WB")  = WB_stab;
Stability(BrainType=="SB")  = SB_stab;
Similarity(BrainType=="WB") = WB_sim;
Similarity(BrainType=="SB") = SB_sim;

Age_long  = repelem(Age,     2, 1);
Sex_long  = repelem(Sex_raw, 2, 1);
RMSD_long = repelem(RMSD,    2, 1);

%% -------------------------------
% 4) Z-score continuous variables
% -------------------------------

z = @(x) (x - mean(x,'omitnan')) ./ std(x,[],'omitnan');

Stability_z  = z(Stability);
Similarity_z = z(Similarity);
Age_z        = z(Age_long);
RMSD_z       = z(RMSD_long);

% Convert Sex to categorical (0=Male, 1=Female)
Sex_cat = categorical(Sex_long, [0 1], {'Male','Female'});

%% -------------------------------
% 5) Build analysis table 
% -------------------------------

tbl = table(Subject, Site, BrainType, Stability_z, Similarity_z, Age_z, Sex_cat, RMSD_z, ...
    'VariableNames', {'Subject','Site','BrainType','Stability','Similarity','Age','Sex','RMSD'});

% Set reference level for BrainType (WB is baseline)
tbl.BrainType = reordercats(tbl.BrainType, ["WB","SB"]);

%% -------------------------------
% 6) Fit linear mixed-effects model (REML)
% -------------------------------

form = ['Stability ~ Similarity*BrainType + Age + Sex + RMSD ' ...
        '+ (Similarity|Subject) + (1|Site)'];

lme = fitlme(tbl, form, 'FitMethod','REML');



