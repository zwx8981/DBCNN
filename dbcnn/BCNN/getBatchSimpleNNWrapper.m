% -------------------------------------------------------------------------
function fn = getBatchSimpleNNWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb, batch) getBatchSimpleNN(imdb, batch, opts) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatchSimpleNN(imdb, batch, opts)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = imdb_get_batch_bcnn(images, opts, ...
                            'prefetch', nargout == 0);
labels = imdb.images.label(batch) ;
numAugments = size(im{1},4)/numel(batch);

labels = reshape(repmat(labels, numAugments, 1), 1, size(im{1},4));
