%% Transfer Training Using Augmented VGG16.

% Load VGG16 Model.
Model = vgg16;

%% Loading The Data Store Along With The Augmented Data Store For Training. 

% Load Image Training Data. 
Folder_Data = './...';
% Name Desired Categories / Labels. 
Categories = {'...', '...', '...'};
% Create Data Store For Training Images. 
Data_Store = imageDatastore(fullfile(Folder_Data, Categories), 'LabelSource', 'foldernames');
% Count Folders (Categories) Found Within The Directory Given & Display it.
Table = countEachLabel(Data_Store);
disp(Table)
% Resize Images To Fit The Input Requirements By The Pre Trained Model. 
Data_Store.ReadFcn = @(filename)BatchResize(filename);
% Split Dataset Randomly From Data Store. 
[Training_Set, Validation_Set] = splitEachLabel(Data_Store, 0.70 , 'randomized');
% Display Training & Validation Values. 
countEachLabel(Training_Set)
countEachLabel(Validation_Set)
% Retrieve Output Layers From The Dataset & Initialize The Wanted Classes. 
Layers_Transfer = Model.Layers(1:end-3);
Class_Number = 3; 
% Initialize A Layer Variable With The Desired Parameters. 
Layers = [
    Layers_Transfer
    fullyConnectedLayer(Class_Number,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
% Set Image Augmentation Options For Input. 
Pixel_Range = [-30 30];
Scale_Range = [0.9 1.1];
Image_Augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',Pixel_Range, ...
    'RandYTranslation',Pixel_Range, ...
    'RandXScale',Scale_Range, ...
    'RandYScale',Scale_Range);
Input_Size = Model.Layers(1).InputSize; 
% Create Augmented Training Images With Parameters Set Above.
Augmented_Train = augmentedImageDatastore(Input_Size(1:2),Training_Set, ...
    'DataAugmentation',Image_Augmenter);
disp(Augmented_Train.NumObservations) 
% Create Augmented Validation Images With Parameters Set Above. 
Augmented_Validation = augmentedImageDatastore(Input_Size(1:2),Validation_Set);
disp(Augmented_Validation.NumObservations) 
% Initialize Training Options For Transfer Learning. 
Mini_Batch_Size = 10;
Options = trainingOptions('sgdm', ...
    'MiniBatchSize',Mini_Batch_Size, ...
    'MaxEpochs', 11, ...
    'InitialLearnRate', 3e-5, ...
    'ValidationData',Augmented_Validation, ...
    'Verbose',false, ...
    'Plots','training-progress');
% Analyze Model. 
analyzeNetwork(Model)

%% Training The Data Model & Saving The Data Model.

% Call Network Training & Save The Data For Main Program Usage. 
VGG16_Model = trainNetwork(Augmented_Train,Layers,Options);
save('VGG16_Model.mat','VGG16_Model')

%% Classify The Validation Images Using The Fine-Tuned Network. 

% Calculate The Model Performance Given Validation Set. 
[Prediction_Augmented,Probability_Augmented] = classify(VGG16_Model,Augmented_Validation);

%% Calculate The Classification Accuracy On The Validation Set.

% Display Validation Accuracy From The Calculated Performance. 
AugmentedValidation = Validation_Set.Labels;
Augmented_Accuracy = mean(Prediction_Augmented == AugmentedValidation);
fprintf("The Validation Accuracy Is: %.2f %%\n", Augmented_Accuracy * 100);

%% The Two Sections Below Are Not Required For Training Or Testing Output. 

%% Tesing Images From An Additional Set Of Images.

% If Model Was Already Trained, Use This To Load It Back. 
load('VGG16_Model.mat')
Net = VGG16_Model;
% Create Data Store For Additional Test Images. 
TestDataStore = imageDatastore('./...','IncludeSubfolders',true,'LabelSource','foldernames');
AugTestDataStore = augmentedImageDatastore([224,224], TestDataStore)
ActualNumber = TestDataStore.Labels;
% Calculate The Classification Prediction For The Images.
Prediction = classify(VGG16_Model,AugTestDataStore);
PredictionCorrect = nnz(Prediction == ActualNumber)
FractionCorrect = PredictionCorrect/numel(Prediction)
% Display The Confusion Chart Of The Model. 
confusionchart(TestDataStore.Labels,Prediction)

%% Testing Images 3x6 Subplot.

% If Model Was Already Trained, Use This To Load It Back. 
load('VGG16_Model.mat')
Net = VGG16_Model;
% Set The Name Of The Labels Wanted To Be Displayed. 
Categories = {'...', '...', '...'};
% Initialize Data Store & Call Resize Image File. 
Location = imageDatastore('./...','IncludeSubfolders',true,'LabelSource','foldernames');
Location.ReadFcn = @(filename)BatchResize(filename);
% Check Length Of Images From File. 
Count = length(Location.Files)
% Loop Used To Display Image, Classfication & Accuracy For User's To See. 
for x = 1:Count
    image = readimage(Location,x);
    Prediction = predict(VGG16_Model,image);
    [Confidence,Index] = max(Prediction);
    Label = Categories{Index};
    subplot(3,6,x),imshow(image)
    title(string(Label) + ", " + num2str(100*Confidence) + "%"); 
end