%% Initiate The Created Model, Categories & Desired Webcam Input. 

% Write The Name Of The Saved Data From The ClassifierTrainer File Created.
Data = load('VGG16_Model.mat');
ClassifierModel = Data.VGG16_Model;
% List The Categories Used In Training On The Class Cell Array Below. 
Categories = {'...', '...', '...'};
% If Having Problems Use 'webcamlist' To Find Your Webcam.  
webcam = webcam();

%% Main Facial Recognition Program. 

% Takes Snapshot Of The Image Being Displayed By The Camera. 
image = snapshot(webcam);
% Uses Haar Cascade Program To Derive The Location Of The User's Face. 
detection = vision.CascadeObjectDetector();
% Create A Small Window To Close Program Once Running. 
hWaitbar = waitbar(0, 'Facial Detection/Classifier', 'Name', 'Running Appliccation', 'CreateCancelBtn', 'delete(gcbf)');
% Initiates Loop As Long As An Image Is Being Received. 
imshow(image);
while true 
   % If The Cancel Button Is Clicked, Break Out Of The Program.  
   if ~ishandle(hWaitbar)
       break;
   % Else Find A Singular Face And Classifity Based On The Created Model.   
   else
       image = snapshot(webcam);
       image2 = rgb2gray(image);
       % Retrives The Facial Detection Location From The Given Image. 
       bbox = step(detection, image2);
       % This Is Used To Keep Track Of Faces Found. 
       [NumRows, NumCols] = size(bbox);
       % If Image Was Not Found & Saved To 'bbox' Continue The Program. 
       if isempty(bbox)
           % Safety Purposes
       % Else If Face Location Was Detected, Perform Operations Below. 
       elseif NumRows == 1 
           % Crop Image Around The Face Detection Found By Haar Cascade. 
           image3 = imcrop(image, bbox);
           % Resize The Image To The Correct Inputs Received By The CNN. 
           resize = imresize(image3, [224 224]);
           % Request Model Prediction To Be Derived Alongside The Accuracy. 
           Prediction = predict(ClassifierModel, resize);
           [Confidence,Index] = max(Prediction);
           % Retrived The Prediction Label. 
           Label = Categories{Index};
           % If Prediction Confidence Is Below 95%, The User Is Unknown.  
           if Confidence * 100 < 95
               % Draws A Red Box Around User's Face Saying Unknown. 
               picture = insertObjectAnnotation(image, 'rectangle', bbox, "Unknown ", 'Color', 'red', 'FontSize', 18);
               imshow(picture)
           % If Prediction Confidence Above 95%, User Will Be Accepted.
           else 
               % Draws A Green Box Alongside The User's Name Above It. 
               picture = insertObjectAnnotation(image, 'rectangle', bbox,  Label + " " + string(Confidence * 100), 'Color', 'Green', 'FontSize', 18);
               imshow(picture)
           end
       % Else If Multiple Faces Are Found, Perform The Operation Below. 
       else 
           % Create A Loop Based On The Number Of Faces Found. 
           for i = 1 : NumRows 
               image3 = imcrop(image, bbox(i,:));
               resize = imresize(image3, [224 224]);
               Prediction = predict(ClassifierModel, resize);
               [Confidence,Index] = max(Prediction);
               % Save The Classification Within A Caell Array.
               MultyFace{i, 1} = Categories{Index};
               % Save The Accuracy Of The Confidence On Another. 
               MultyAcc{i, 1} = Confidence * 100;
           end
           % Create A Structure For Comparison.
           [NumRowsM, NumColsM] = size(MultyFace);
           % Compare The Single & Multy Classification. 
           if NumRowsM > NumRows
               MultyFace(end, :) = [];
               MultyAcc(end, :) = [];
           end
           % Draw A White Border Around All Faces Alongside Their Accuracy.
           picture = insertObjectAnnotation(image, 'rectangle', bbox,  MultyFace + " " + string(MultyAcc), 'Color', 'White', 'FontSize', 18);
           imshow(picture)
       end
   end
end
