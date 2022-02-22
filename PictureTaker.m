%% Application Used To Take 100 Pictures Of The User's Face For Training. 

% If Having Problems Use 'webcamlist' To Find Your Webcam.  
webcam = webcam();
% Use Your Desired File Path. 
savepath = '...'; 
% Images Will Be Saved In Order, No Need To Change This. 
nametemplate = 'image_%04d.jpg';
% Uing Haar Cascade Program To Find User's Face. 
detection = vision.CascadeObjectDetector();
imnum = 0;
% Loop Used To Take The 100 Pictures. 
for i = 1 : 100
    % Retrieve Screen & Face Detection Coordinates. 
    image = snapshot(webcam);
    image2 = rgb2gray(image);
    bbox = step(detection, image2);
    [NumRows, NumCols] = size(bbox);
    % If No Face Or More Than One Are Found, Iterate One More Time.
    if isempty(bbox) || NumRows > 1 
           i = i - 1;
    % Else, Save The Retrieved Image As Directed.        
    else
        % Crop The Face Found By Haar Function & Display It. 
        x2 = imcrop(image, bbox);
        imshow(x2)
        % Increse Number For Image Naming. 
        imnum = imnum + 1;
        % Save The Image Given Directory, Path & Name. 
        thisfile = sprintf(nametemplate, imnum);  
        fullname = fullfile(savepath, thisfile);  
        imwrite(x2, fullname);  
    end
end
