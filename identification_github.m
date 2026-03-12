%% Fingerprinting and Discriminability
% ------------------------------------------------------------
% Computes fingerprinting accuracy and discriminability
% between Movie and Rest FC vectors for WB and SB
%
% Input: 
%   all_results (cell array where all_results{s}.out contains FC vectors)
%
% Output: 
%   results_fingerprint 
%
% Required fields in all_results{s}.out:
%   WB:
%       FC_z_vec_scrub_Movie
%       FC_z_vec_scrub_Rest
%   SB:
%       SBFC_z_vec_scrub_Movie
%       SBFC_z_vec_scrub_Rest
% ------------------------------------------------------------

S = numel(all_results);

configs = { ...
    struct('brain_type', 'WB', 'fieldA', 'FC_z_vec_scrub_Movie',  'fieldB', 'FC_z_vec_scrub_Rest'), ...
    struct('brain_type', 'SB', 'fieldA', 'SBFC_z_vec_scrub_Movie', 'fieldB', 'SBFC_z_vec_scrub_Rest') ...
    };

results_fingerprint = struct( ...
    'brain_type', {}, ...
    'n_subjects_total', {}, ...
    'n_subjects_used', {}, ...
    'n_edges', {}, ...
    'accuracy_Movie_to_Rest', {}, ...
    'accuracy_Rest_to_Movie', {}, ...
    'discr_Movie_to_Rest', {}, ...
    'discr_Rest_to_Movie', {}, ...
    'discr_sym', {} );

for c = 1:numel(configs)
    cfg = configs{c};

    % --------------------------------------------------------
    % Extract subjects 
    % --------------------------------------------------------
    vec_length = numel(all_results{1}.out.(cfg.fieldA));
    A = [];
    B = [];

    for s = 1:S
        out = all_results{s}.out;
        A(:, end+1) = out.(cfg.fieldA)(:);
        B(:, end+1) = out.(cfg.fieldB)(:); 
    end
    n_used = size(A,2);
    n_edges = size(A, 1);

    fprintf('\n[%s] Using %d/%d subjects, edges = %d\n', ...
        cfg.brain_type, n_used, S, n_edges);

    % --------------------------------------------------------
    % Similarity matrices
    % --------------------------------------------------------
    Sim_AB = corr(A, B, 'Type', 'Pearson', 'Rows', 'pairwise'); % Movie -> Rest
    Sim_BA = corr(B, A, 'Type', 'Pearson', 'Rows', 'pairwise'); % Rest  -> Movie

    % --------------------------------------------------------
    % Fingerprinting accuracy
    % --------------------------------------------------------
    [~, best_AB] = max(Sim_AB, [], 2);
    [~, best_BA] = max(Sim_BA, [], 2);

    true_labels = (1:n_used)';

    id_success_AB = (best_AB == true_labels);
    id_success_BA = (best_BA == true_labels);

    acc_AB = 100 * mean(id_success_AB, 'omitnan');
    acc_BA = 100 * mean(id_success_BA, 'omitnan');

    % --------------------------------------------------------
    % Discriminability
    % --------------------------------------------------------
    discr_AB = NaN(n_used,1);
    for i = 1:n_used
        self_val = Sim_AB(i,i);
        others = Sim_AB(i,:);
        others(i) = NaN;
        discr_AB(i) = mean(others < self_val, 'omitnan');
    end

    discr_BA = NaN(n_used,1);
    for i = 1:n_used
        self_val = Sim_BA(i,i);
        others = Sim_BA(i,:);
        others(i) = NaN;
        discr_BA(i) = mean(others < self_val, 'omitnan');
    end

    group_discr_AB  = mean(discr_AB, 'omitnan');
    group_discr_BA  = mean(discr_BA, 'omitnan');
    group_discr_sym = mean([group_discr_AB, group_discr_BA], 'omitnan');

    fprintf('[%s] Accuracy: Movie->Rest = %.2f%% | Rest->Movie = %.2f%%\n', ...
        cfg.brain_type, acc_AB, acc_BA);
    fprintf('[%s] Discriminability: Movie->Rest = %.4f | Rest->Movie = %.4f | Sym = %.4f\n', ...
        cfg.brain_type, group_discr_AB, group_discr_BA, group_discr_sym);

    % --------------------------------------------------------
    % Store results
    % --------------------------------------------------------
    results_fingerprint(end+1) = struct( ...
        'brain_type', cfg.brain_type, ...
        'n_subjects_total', S, ...
        'n_subjects_used', n_used, ...
        'n_edges', n_edges, ...
        'accuracy_Movie_to_Rest', acc_AB, ...
        'accuracy_Rest_to_Movie', acc_BA, ...
        'discr_Movie_to_Rest', group_discr_AB, ...
        'discr_Rest_to_Movie', group_discr_BA, ...
        'discr_sym', group_discr_sym );
end

