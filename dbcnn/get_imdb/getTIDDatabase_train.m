function imdb = getTIDDatabase_train(Dir)

fileID = fopen(fullfile(Dir,'\mos_with_names.txt'));
mos_name = textscan(fileID,'%f %s');
mos = mos_name{1,1};
name = mos_name{1,2};

path = cell(1,3000);
for i = 1:3000
    namet = name{i,1};
%     path{1,i} = fullfile(Dir,'distorted_images',namet); 
    path{1,i} = fullfile('distorted_images',namet); 
end

imdb.mos = mos';
imdb.label = imdb.mos;
imdb.name = name;
imdb.imgpath = path;
imdb.filenum = 3000;
imdb.refnum = 25;
imdb.dataset = 'TID2013';

refpath = fullfile(Dir,'reference_images');
refpath = strcat(refpath,'\');
dir_rf = dir([refpath '*.BMP']);
imdb.refname = cell(1,25);
for i = 1:25
    file_name = dir_rf(i).name;
    %imdb.refname{i} = string(file_name);
    imdb.refname{i} = file_name;
end

imdb.refnames_all = cell(1,3000);
for i = 1:25
    for j = 1:120
        imdb.refnames_all{1,(i-1)*120+j} = imdb.refname{i};
    end
end

sel = randperm(25);
train_path = [];
train_mos = [];
for i = 1:20
    train_sel = strcmpi(imdb.refname(sel(i)),imdb.refnames_all );
    train_sel = find(train_sel == 1);
    train_path = [train_path, imdb.imgpath(train_sel)]; 
    train_mos = [train_mos,imdb.label(train_sel)];
end

test_path = [];
test_mos = [];
for i = 21:25
    test_sel = strcmpi(imdb.refname(sel(i)),imdb.refnames_all );
    test_sel = find(test_sel == 1);
    test_path = [test_path, imdb.imgpath(test_sel)]; 
    test_mos = [test_mos,imdb.label(test_sel)];
end

imdb.images.id = 1:3000 ;
imdb.images.set = [ones(1,size(train_path,2)),2*ones(1,size(test_path,2))];
imdb.images.label = [train_mos,test_mos];
imdb.images.label = imdb.images.label * 100/9 ;
imdb.classes.description = {'TID2013'};
imdb.images.name = [train_path,test_path] ;
imdb.imageDir = Dir ;
fclose(fileID);
