function print_dataset_info(imdb)
multiLabel = (size(imdb.images.label, 1) > 1) ;
train = ismember(imdb.images.set, [1 2]) ;
test = ismember(imdb.images.set, [3]) ;
fprintf('dataset: classes: %d in use. These are:\n', sum(imdb.meta.inUse)) ;
trainIm = ismember(imdb.images.id, imdb.images.id(train)) ;
testIm = ismember(imdb.images.id, imdb.images.id(test)) ;
for i = find(imdb.meta.inUse)
  if ~multiLabel
    a = sum(imdb.images.label(trainIm) == i) ;
    b = sum(imdb.images.label(testIm) == i) ;
    c = sum(imdb.images.label == i) ;
  else
    a = sum(imdb.images.label(i, trainIm) > 0, 2) ;
    b = sum(imdb.images.label(i, testIm) > 0, 2) ;
    c = sum(imdb.images.label(i, :) > 0, 2) ;
  end
  fprintf('%4d: %15s (train: %5d, test: %5d total: %5d)\n', ...
    i, imdb.meta.classes{i}, a, b, c) ;
end
a = numel(trainIm) ;
b = numel(testIm) ;
c = numel(imdb.images.id) ;
fprintf('%4d: %15s (train: %5d, test: %5d total: %5d)\n', ...
  +inf, '**total**', a, b, c) ;
fprintf('dataset: there are %d images (%d trainval %d test)\n', ...
  numel(imdb.images.id), sum(train), sum(test)) ;