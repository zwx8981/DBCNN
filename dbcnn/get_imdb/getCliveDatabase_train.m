function imdb = getCliveDatabase_train(Dir)

imdb.imgpath = cell(1,1162);
imgpath = fullfile(Dir,'Data','AllImages_release.mat');
img = load(imgpath);
img = img.AllImages_release;

mospath = fullfile(Dir,'Data','AllMOS_release.mat');
mos = load(mospath);
mos = mos.AllMOS_release;
imdb.mos = mos(8:end);

for i = 8:1169
    file_name = img{i,1};     
    imdb.imgpath{i-7} = fullfile('Images',file_name);
end

sel = randperm(1162);
train_sel = sel(1:round(0.8*1162));
test_sel = sel(round(0.8*1162)+1:end);

train_path = imdb.imgpath(train_sel);
test_path = imdb.imgpath(test_sel);

train_mos = imdb.mos(train_sel);
test_mos = imdb.mos(test_sel);

imdb.images.id = 1:1162 ;
imdb.images.set = [ones(1,size(train_path,2)),2*ones(1,size(test_path,2))];
imdb.images.label = [train_mos,test_mos];

imdb.classes.description = {'LIVE_CHAN'};
imdb.images.name = [train_path,test_path] ;
imdb.imageDir = Dir ;
% fclose(fileID);