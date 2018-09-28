function imdb = getMLIVEDatabase_train(Dir)


Scores1 = load([Dir, '\Part 1\Scores.mat']);
imdb.dmos1 = Scores1.DMOSscores;
Scores2 = load([Dir, '\Part 2\Scores.mat']);
imdb.dmos2 = Scores2.DMOSscores;

Imagelists1 = load([Dir, '\Part 1\Imagelists.mat']);
refnames1 = Imagelists1.refimgs;
distimgs1 = Imagelists1.distimgs;
ref4dist1 = Imagelists1.ref4dist;
Imagelists2 = load([Dir, '\Part 2\Imagelists.mat']);
refnames2 = Imagelists2.refimgs;
distimgs2 = Imagelists2.distimgs;
ref4dist2 = Imagelists2.ref4dist;

imdb.dmos = [imdb.dmos1,imdb.dmos2];

imdb.dir_dist1 = cell(1,225);
for i = 1:225
    imdb.dir_dist1{1,i} = fullfile('Part 1\blurjpeg',distimgs1{i,1});
end
imdb.dir_dist2 = cell(1,225);
for i = 1:225
    imdb.dir_dist2{1,i} = fullfile('Part 2\blurnoise',distimgs2{i,1});
end

imdb.imgDir =  cat(2,imdb.dir_dist1,imdb.dir_dist2);
imdb.dataset = 'MLIVE';
imdb.filenum = 450;

imdb.dmosall = imdb.dmos;
imdb.refnum = 15;
for i = 1:15
    imdb.refname{i} = num2str(i);
end
for i = 1:225
    imdb.refnames_all{i} = num2str(ref4dist1(i));
end
for i = 226:450
    imdb.refnames_all{i} = num2str(ref4dist2(i-225));
end

sel = randperm(15);
% sel = 1:15;
train_path = [];
train_dmos = [];
for i = 1:12
    train_sel = strcmpi(imdb.refname(sel(i)),imdb.refnames_all);
    train_sel = find(train_sel == 1);
    train_path = [train_path, imdb.imgDir(train_sel)]; 
    train_dmos = [train_dmos,imdb.dmos(train_sel)];
end

test_path = [];
test_dmos = [];
for i = 13:15
    test_sel = strcmpi(imdb.refname(sel(i)),imdb.refnames_all);
    test_sel = find(test_sel == 1);
    test_path = [test_path, imdb.imgDir(test_sel)]; 
    test_dmos = [test_dmos,imdb.dmos(test_sel)];
end

imdb.images.id = 1:450 ;
imdb.images.set = [ones(1,size(train_path,2)),2*ones(1,size(test_path,2))];
imdb.images.label = [train_dmos,test_dmos];
imdb.classes.description = {'MLIVE'};
imdb.images.name = [train_path,test_path] ;
imdb.imageDir = Dir;
