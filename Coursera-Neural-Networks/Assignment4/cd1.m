function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    visible_data = sample_bernoulli(visible_data);
    
    cases = size(visible_data, 2);
    data_gradient = zeros(size(rbm_w));
    reconstruction_gradient = zeros(size(rbm_w));
    
    visible = visible_data;
    hidden = sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w, visible));
    reconstruction_visible = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w, hidden));
    %reconstruction_hidden = sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w, reconstruction_visible));
    reconstruction_hidden = visible_state_to_hidden_probabilities(rbm_w, reconstruction_visible);
    for i = 1:cases
        data_gradient = data_gradient + (hidden(:, i) * visible(:, i)');
        reconstruction_gradient = reconstruction_gradient + (reconstruction_hidden(:, i) * reconstruction_visible(:, i)');
    end
    ret = (data_gradient ./ cases) - (reconstruction_gradient ./ cases);
end