%% Function Used To Resize Images To The Correct Size. 

function Iout = readAndPreprocessImage(filename)
I = imread(filename);
% If One Of The Images Are Grayscale, We Will Turn The Pictures Into A RGB 
% Image By Replicating Them 3 Times. 
if ismatrix(I)
    I = cat(3,I,I,I);
end
% Resizing The Image To The Input Allowed By The CNN.
Iout = imresize(I, [224 224]);
end

