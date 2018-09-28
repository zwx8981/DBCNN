% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = imdb_get_batch_bcnn(images, opts, ...
                            'prefetch', nargout == 0);
labels = imdb.images.label(batch) ;
numAugments = size(im{1},4)/numel(batch);

labels = reshape(repmat(labels, numAugments, 1), 1, size(im{1},4));

if nargout > 0
  if useGpu
    im1 = gpuArray(im{1}) ;
    im2 = gpuArray(im{2}) ;
  else
      im1 = im{1};
      im2 = im{2};
  end
  inputs = {'input', im1, 'netb_input', im2, 'label', labels} ;
end

