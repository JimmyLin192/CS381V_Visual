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
INL_ITERATIONS = 150;
INL_THRESHOLD = 3;
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
    % Find the nearest neighbor descriptor in im2 for some random descriptor
    % from im1:

    allMatchMatrix = [];
    TNNmatchMatrix = [];
    TRTmatchMatrix = [];
    for im1_des_index = 1:size(d1,2)
        aSingleSIFTDescriptor = d1(:, im1_des_index);
        dists = dist2(double(aSingleSIFTDescriptor)', double(d2)');
        [sortedDists, sortedIndices] = sort(dists, 'ascend');
        match_info = [im1_des_index; sortedIndices(1); sortedDists(1)];
        allMatchMatrix = [allMatchMatrix match_info];
        % ratio tests
        if (sortedDists(1) <= TRT_THRESHOLD * sortedDists(2))
            TRTmatchMatrix = [TRTmatchMatrix match_info];
        end
    end
    mean_dist = mean(allMatchMatrix(3, :));
    threshold = TNN_THRESHOLD * mean_dist;
    for match_index = 1:size(allMatchMatrix,2)
        if (allMatchMatrix(3,match_index) <= threshold)
            TNNmatchMatrix = [TNNmatchMatrix allMatchMatrix(:,match_index)];
        end
    end

    INLmatchMatrix = [];
    max_inliers = 0;
    for INL_ITER = 1:INL_ITERATIONS
        randomIndices = randperm(n1);
        randomIndices_d2 = randperm(n2);
        threeSIFTDescriptors = d1(:, randomIndices(1:3));
        %trans_template_frame_pos = [];
        %for index = 1 : 3
        %    dists = dist2(double(threeSIFTDescriptors(:,index))', double(d2)');
        %    [sortedDists, sortedIndices] = sort(dists, 'ascend');
        %    trans_template_frame_pos = [trans_template_frame_pos f2(1:2, sortedIndices(1))];
        %end
        scene_frame_pos = f1(1:2, randomIndices(1:3));
        trans_template_frame_pos = f2(1:2, randomIndices_d2(1:3));
        tform = fitgeotrans(trans_template_frame_pos', scene_frame_pos', 'affine');
        trans_f1 = transformPointsForward(tform, double(f1(1:2, :))');
        inliers = 0;
        for index = 1 : size(trans_f1, 1)
            dists = dist2(double(trans_f1(index, :)), double(f2(1:2, :))');
            min_dist = min(dists);
            if (min_dist <= INL_THRESHOLD)
                inliers = inliers + 1;
            end
        end
        if (inliers > max_inliers)
            max_inliers = inliers;
            best_transform = tform;
        end
    end
    trans_f1 = transformPointsForward(best_transform, f1(1:2, :)');
    for index = 1 : size(trans_f1, 1)
        dists = dist2(double(trans_f1(index, :)), double(f2(1:2, :))');
        [sortedDists, sortedIndices] = sort(dists, 'ascend');
        if (sortedDists(1) <= INL_THRESHOLD)
            match_info = [index; sortedIndices(1); sortedDists(1)];
            INLmatchMatrix = [INLmatchMatrix match_info];
        end
    end
    % We have just one match here, but to use the display functions below, you
    % can simply expand this matrix to include one column for each match.
    numMatches = size(TNNmatchMatrix,2);
    
    % Display the matched patch
    clf;
    showMatchingPatches(TNNmatchMatrix, d1, d2, f1, f2, im1, im2, SHOW_ALL_MATCHES_AT_ONCE);
    fprintf('Showing an example of %s,TNN patch match. \n', scenenames{scenenum});
    keyboard;
    
    % An alternate display - show lines connecting the matches (no patches)
    clf;
    showLinesBetweenMatches(im1, im2, f1, f2, TNNmatchMatrix);
    fprintf('Showing the %s,TNN matches with lines connecting.\n', scenenames{scenenum});
    keyboard;

    % Display the matched patch
    clf;
    showMatchingPatches(TRTmatchMatrix, d1, d2, f1, f2, im1, im2, SHOW_ALL_MATCHES_AT_ONCE);
    fprintf('Showing an example of %s:TRT patch match. \n', scenenames{scenenum});
    keyboard;
    
    % An alternate display - show lines connecting the matches (no patches)
    clf;
    showLinesBetweenMatches(im1, im2, f1, f2, TRTmatchMatrix);
    fprintf('Showing the %s:TRT matches with lines connecting.\n', scenenames{scenenum});
    keyboard;

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
end
