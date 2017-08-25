X = zeros(1000,128,128);
y = zeros(1000,1);
  folder = '/home/mridul/machine_learning_course/English/Fnt/Sample001';
  filePattern = fullfile(folder, 'img*.png');
  fileNames = dir(filePattern);

  for k = 1:1000
    fullFileName = fullfile(folder, fileNames(k).name);
    X(k,:,:) = imread(fullFileName);
  end

input_layer_size  = 16384;
num_labels = 62;

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

pred = predictOneVsAll(all_theta, X);
