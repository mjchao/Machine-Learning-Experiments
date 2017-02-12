function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
    total_energy = 0;
    configs = size(visible_state, 2);
    for i = 1:configs
        visible = visible_state(:, i); % V x 1
        hidden = hidden_state(:, i);   % H x 1
        energy_matrix = (hidden * visible') .* rbm_w;
        energy = sum(energy_matrix(:))
        total_energy = total_energy + energy;
    end
    G = total_energy / configs;
end
