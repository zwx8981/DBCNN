function X = randnd(beta,varargin)
%% X = randnd(beta,varargin)
% This function is a "decorator" for the Matlab in-build 'randn' function.
% It takes an additional first argument (beta) which shapes the spectral
% characteristic of the data (in all dimensions) as f^beta. The output data
% is scaled to zero mean and unit standard deviation.
%
% For help on 'varargin' go to Matlab help for the 'randn' function.
%
% EXAMPLE 1 - Input robustness
% A = randnd(-1,[2^5+1 1]); % Handle odd dimensionality
% A = randnd((rand-0.5)*4,[134 12 1 1 1 26 1 9]); % Handle large number of dimentions
% A = randnd(-1,[1 0 3]); % Handle 0's
% A = randnd(2,[1 1 1 256]); % Handle leading singleton dimensions
%
% EXAMPLE 2 - Pink map
% A = randnd(-1,2^8); % Generate 256 x 256 grid with random 1/f (pink) noise
% contourf(A,'edgecolor','none'); axis square; % Display it as a contour plot
%
% EXAMPLE 3 - Brownian cube
% A = randnd(-2,[2^4 2^4 2^4]); % Generate 16 x 16 x 16 cube of f^-2 (Brownian) noise
% [X,Y,Z] = meshgrid(1:2^4); % Generate 3d meshgrid
% scatter3(X(:),Y(:),Z(:),100,A(:),'filled');
% axis vis3d; axis off;
%
% EXAMPLE 4 - Single blue noise
% p = single(magic(10)); % Single precision matrix of interest
% A = randnd(1,size(p),'like',p); % Generate f (blue) noise 'like' p
% isa(A,'single') % X is also a single precision random number
%
% EXAMPLE 5 - Proof that beta scales power spectral density correctly
% beta = -2;
% A = randnd(beta,[1 2^11]);
% Afft = abs(fft(A.^2)); % Get the magnitude of power (i.e. signal^2) spectrum
% Afft = Afft(2:length(Afft)/2); % Get the spectrum between DC and Nyquist
% k = 1:length(Afft); % Create wavenumber (frequency) vector
% P = polyfit(log(k),log(Afft),1); % Fit a straight line to the loglog plot
% loglog(k,Afft,'.',k,exp(P(2))*k.^P(1),'-'); grid on; % Plot on loglog axis
% xlim([min(k) max(k)]); title(['Beta = ',num2str(beta),'; Slope = ',num2str(P(1))]);
%
%
% Based on similar functions by Jon Yearsley and Hristo Zhivomirov
% Written by Marcin Konowalczyk
% Timmel Group @ Oxford University
 
%% Parse the input
narginchk(0,Inf); nargoutchk(0,1);
 
if nargin < 2 || isempty(beta); beta = 0; end % Default to white noise
assert(isnumeric(beta) && isequal(size(beta),[1 1]),'''beta'' must be a number');
assert(-2 <= beta && beta <= 2,'''beta'' out of range'); % Put on reasonable bounds
 
%% Generate N-dimensional white noise with 'randn'
X = randn(varargin{:});
if isempty(X); return; end; % Usually happens when size vector contains zeros
 
% Squeeze prevents an error if X has more than one leading singleton dimension
% This is a slight deviation from the pure functionality of 'randn'
X = squeeze(X);
 
% Return if white noise is requested
if beta == 0; return; end;
 
%% Generate corresponding N-dimensional matrix of multipliers
N = size(X);
% Create matrix of multipliers (M) of X in the frequency domain
M = []; 
for j = 1:length(N)
    n = N(j);
    
    if (rem(n,2)~=0) % if n is odd
        % Nyquist frequency bin does not show up in odd-numbered fft
        k = ifftshift(-(n-1)/2:(n-1)/2);
    else
        k = ifftshift(-n/2:n/2-1);
    end
    
    % Spectral multipliers
    m = (k.^2)';
    
    if isempty(M);
        M = m;
    else
        % Create the permutation vector
        M_perm = circshift(1:length(size(M))+1,1,2);
        % Permute a singleton dimension to the beginning of M
        M = permute(M,M_perm);
        % Add m along the first dimension of M
        M = bsxfun(@plus,M,m);
    end
end
clear j M_perm k n m
% Reverse M to match X (since new dimensions were being added form the left)
M = permute(M,length(size(M)):-1:1);
assert(isequal(size(M),size(X)),'Bad programming error'); % This should never occur
 
% Shape the amplitude multipliers by (beta/2) which corresponds to shaping
% the power by beta
M = M.^(beta/2);
 
% Set the DC component to zero
M(1,1) = 0;
 
%% Multiply X by M in frequency domain
Xstd = std(X(:));
Xmean = mean(X(:));
X = real(ifftn(fftn(X).*M));
 
% Force zero mean unity standard deviation
X = X - mean(X(:));
X = X./std(X(:));
 
% Restore the standard deviation and mean from before the spectral shaping.
% This ensures the random sample from randn is truly random. After all, if
% the mean was always exactly zero it would not be all that random.
X = X + Xmean;
X = X.*Xstd;
end