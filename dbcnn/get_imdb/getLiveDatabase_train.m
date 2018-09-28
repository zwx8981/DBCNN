function imdb = getLiveDatabase_train(Dir)

refpath = fullfile(Dir,'refimgs');
refpath = strcat(refpath,'\');
dir_rf = dir([refpath '*.bmp']);
dmos_t = load(fullfile(Dir,'dmos_realigned.mat'));
imdb.dmos = dmos_t.dmos_new;
imdb.orgs = dmos_t.orgs;

refname = load(fullfile(Dir,'refnames_all.mat'));
imdb.refnames_all = refname.refnames_all;

imdb.j2dmos = imdb.dmos(1:227);
imdb.jpdmos = imdb.dmos(228:460);
imdb.wndmos = imdb.dmos(461:634);
imdb.gbdmos = imdb.dmos(635:808);
imdb.ffdmos = imdb.dmos(809:end);

imdb.j2orgs = imdb.orgs(1:227);
imdb.jporgs = imdb.orgs(228:460);
imdb.wnorgs = imdb.orgs(461:634);
imdb.gborgs = imdb.orgs(635:808);
imdb.fforgs = imdb.orgs(809:end);

imdb.orgs = [imdb.j2orgs,imdb.jporgs,imdb.wnorgs,imdb.gborgs,imdb.fforgs];

imdb.refname = cell(1,29);
for i = 1:29
    file_name = dir_rf(i).name;
    %imdb.refname{i} = string(file_name);
    imdb.refname{i} = file_name;
end


%%jp2k
index = 1;
imdb.dir_j2 = cell(1,227);
for i = 1:227
    file_name = strcat('img',num2str(i),'.bmp');
    imdb.dir_j2{index} = fullfile('jp2k',file_name);
    index = index + 1;
end

%%jpeg
index = 1;
imdb.dir_jp = cell(1,233);
for i = 1:233
    file_name = strcat('img',num2str(i),'.bmp');
    imdb.dir_jp{index} = fullfile('jpeg',file_name);
    index = index + 1;
end

%%white noise
index = 1;
imdb.dir_wn = cell(1,174);
for i = 1:174
       file_name = strcat('img',num2str(i),'.bmp');
       imdb.dir_wn{index} = fullfile('wn',file_name);
       index = index + 1;
end

%%gblur
index = 1;
imdb.dir_gb = cell(1,174);
for i = 1:174
    file_name = strcat('img',num2str(i),'.bmp');
    imdb.dir_gb{index} = fullfile('gblur',file_name);
    index = index + 1;
end

%%fast fading
index = 1;
imdb.dir_ff = cell(1,174);
for i = 1:174
    file_name = strcat('img',num2str(i),'.bmp');
    imdb.dir_ff{index} = fullfile('fastfading',file_name);
    index = index + 1;
end

imdb.imgpath =  cat(2,imdb.dir_j2,imdb.dir_jp,imdb.dir_wn,imdb.dir_gb,imdb.dir_ff);
imdb.dataset = 'LIVE';
imdb.filenum = 982;

sel = randperm(29);
train_path = [];
train_dmos = [];
for i = 1:23
    train_sel = strcmpi(imdb.refname(sel(i)),refname.refnames_all);
    train_sel = train_sel.*(~imdb.orgs);
    train_sel = find(train_sel == 1);
    train_path = [train_path, imdb.imgpath(train_sel)]; 
    train_dmos = [train_dmos,imdb.dmos(train_sel)];
end

test_path = [];
test_dmos = [];
for i = 24:29
    test_sel = strcmpi(imdb.refname(sel(i)),refname.refnames_all);
    test_sel = test_sel.*(~imdb.orgs);
    test_sel = find(test_sel == 1);
    test_path = [test_path, imdb.imgpath(test_sel)]; 
    test_dmos = [test_dmos,imdb.dmos(test_sel)];
end

imdb.images.id = 1:779 ;
imdb.images.set = [ones(1,size(train_path,2)),2*ones(1,size(test_path,2))];
imdb.images.label = [train_dmos,test_dmos];
imdb.classes.description = {'LIVE'};
imdb.images.name = [train_path,test_path] ;
% fclose(fileID);
% 
% imdb.dmosall = imdb.dmos;
% imdb.dmos = [imdb.j2dmos,imdb.jpdmos,imdb.wndmos,imdb.gbdmos,imdb.ffdmos];
% imdb.refnum = 29;
% imdb.refnameall = refname.refnames_all;

