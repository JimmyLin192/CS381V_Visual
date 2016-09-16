%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CS381V Visual Recognition @ UT Austin
%% NAME: Xin Lin, EID: XL5224
%% Prof. Kristen Grauman
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% See http://www.vlfeat.org/install-matlab.html
fprintf('Be sure to add VLFeat path.\n');

clear;
close all;

% THRESHOLD
TNN_THRESHOLD = 0.8;
TRT_THRESHOLD = 0.6;
INL_ITERATIONS = 300;
INL_THRESHOLD = 50;
DETECT_THRESHOLD = 10;
fprintf('Threshold for Thresholed nearest neigbhour: %f.\n ', TNN_THRESHOLD)
fprintf('Threshold for Thresholed ratio test: %f.\n ', TRT_THRESHOLD)
fprintf('Number of Iterations for INLIERS CHECK: %d.\n ', INL_ITERATIONS)
fprintf('Threshold for INLIERS CHECK: %f.\n ', INL_THRESHOLD)

% Some flags
DISPLAY_PATCHES = 1;
SHOW_ALL_MATCHES_AT_ONCE = 1;


% Constants
N = 50;  % how many SIFT features to display for visualization of features


templatename = 'object-template.jpg';
scenenames = {'object-template-rotated.jpg', 'scene1.jpg', 'scene2.jpg'};


% Read in the object template image.  This is the thing we'll search for in
% the scene images.
im1 = im2single(rgb2gray(imread(templatename)));


% Extract SIFT features from the template image.
%
% 'f' refers to a matrix of "frames".  It is 4 x n, where n is the number
% of SIFT features detected.  Thus, each column refers to one SIFT descriptor.  
% The first row gives the x positions, second row gives the y positions, 
% third row gives the scales, fourth row gives the orientations.  You will
% need the x and y positions for this assignment.
%
% 'd' refers to a matrix of "descriptors".  It is 128 x n.  Each column 
% is a 128-dimensional SIFT descriptor.
%
% See VLFeats for more details on the contents of the frames and
% descriptors.
[f1, d1] = vl_sift(im1);



% count number of descriptors found in im1
n1 = size(d1,2);


% Loop through the scene images and do some processing
for scenenum = 1:length(scenenames)
    
    fprintf('Reading image %s for the scene to search....\n', scenenames{scenenum});
    im2 = im2single(rgb2gray(imread(scenenames{scenenum})));
    
    
    % Extract SIFT features from this scene image
    [f2, d2] = vl_sift(im2);
    n2 = size(d2,2);
    
    % Show a random subset of the SIFT patches for the two images
    if(DISPLAY_PATCHES)
        
        displayDetectedSIFTFeatures(im1, im2, f1, f2, d1, d2, N);
        
        fprintf('Showing a random sample of the sift descriptors.  Type dbcont to continue.\n');
        
        keyboard;
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    allMatchMatrix = [];
    for im1_des_index = 1:size(d1,2)
        aSingleSIFTDescriptor = d1(:, im1_des_index);
        dists = dist2(double(aSingleSIFTDescriptor)', double(d2)');
        [sortedDists, sortedIndices] = sort(dists, 'ascend');
        match_info = [im1_des_index; sortedIndices(1); sortedDists(1)];
        allMatchMatrix = [allMatchMatrix match_info];
    end
    INLmatchMatrix = [];
    max_inliers = 0;
    for INL_ITER = 1:INL_ITERATIONS
        randomIndices = randperm(size(allMatchMatrix,2));
        randomIndices = randomIndices(1:3);
        im1_des_indices = allMatchMatrix(1, randomIndices);
        im2_des_indices = allMatchMatrix(2, randomIndices);
        trans_template_frame_pos = f1(1:2, im1_des_indices);
        scene_frame_pos = f2(1:2, im2_des_indices);
        % in case of three random descriptors are positionally co-linear 
        try
            tform = fitgeotrans(trans_template_frame_pos', scene_frame_pos', 'affine');
        catch
            continue
        end
        inliers = 0;
        for index = 1 : size(allMatchMatrix, 2)
            im1_des_index = allMatchMatrix(1,index);
            im2_des_index = allMatchMatrix(2,index);
            trans_f1 = transformPointsForward(tform, f1(1:2,im1_des_index)');
            min_dist = dist2(double(trans_f1), double(f2(1:2, im2_des_index))');
            if (min_dist <= INL_THRESHOLD)
                inliers = inliers + 1;
            end
        end
        if (inliers > max_inliers)
            max_inliers = inliers;
            best_transform = tform;
        end
    end
    for index = 1 : size(allMatchMatrix, 2)
        im1_des_index = allMatchMatrix(1,index);
        im2_des_index = allMatchMatrix(2,index);
        trans_f1 = transformPointsForward(best_transform, f1(1:2,im1_des_index)');
        min_dist = dist2(double(trans_f1), double(f2(1:2, im2_des_index))');
        if (min_dist <= INL_THRESHOLD)
            INLmatchMatrix = [INLmatchMatrix allMatchMatrix(:, index)];
        end
    end
    
    % Display the matched patch
    clf;
    showMatchingPatches(INLmatchMatrix, d1, d2, f1, f2, im1, im2, SHOW_ALL_MATCHES_AT_ONCE);
    fprintf('Showing an example of %s:INL patch match. \n', scenenames{scenenum});
    keyboard;
    
    % An alternate display - show lines connecting the matches (no patches)
    clf;
    showLinesBetweenMatches(im1, im2, f1, f2, INLmatchMatrix);
    fprintf('Showing the %s:INL matches with lines connecting.\n', scenenames{scenenum});
    keyboard;
    % made decision and draw the rectangle
    numMatches = size(INLmatchMatrix,2);
    if (numMatches >= DETECT_THRESHOLD) % means present
        w = size(im1, 2);
        h = size(im1, 1);
        corners = [[1 1]; [w 1]; [1, h]; [w h]];
        trans_corners = transformPointsForward(best_transform, corners);
        imshow(im2)
        drawRectangle(trans_corners', 'r');
    end
    keyboard
end
