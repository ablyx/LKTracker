FNames = {'test.jpg';};
        
for p=1 : size(FNames)
    
    % get pic
    pic = double(imread(FNames{p}));
    
    [rows, cols] = size(pic);
    
    gx = pic;
    gy = pic;
    
    % get gx
    % last row of gx is same as pic
    for row=1:rows-1
        for col=1:cols     
            gx(row,col) = pic(row+1,col) - pic(row,col);
        end
    end
    
    % get gy
    % last row of gy is same as pic
    for row=1:rows
        for col=1:cols-1     
            gy(row,col) = pic(row,col+1) - pic(row,col);
        end
    end
    
    I_xx = gx.*gx;
    I_xy = gx.*gy;
    I_yy = gy.*gy;
    
    fullwin = 13;
    gkern = ones(13,13);
    
    W_xx = conv2(I_xx,gkern);
    W_xy = conv2(I_xy,gkern);
    W_yy = conv2(I_yy,gkern);
    
    % the size of output matrix has 12 more rows and 12 more columns
    % compared to the input matrix
    % the relationship is gaussian window size - 1
    
    [wrows, wcols] = size(W_yy);
    
    eig_min = zeros(size(W_yy));
    
    for row=1:wrows
        for col=1:wcols
            Wxx = W_xx(row,col);
            Wxy = W_xy(row,col);
            Wyy = W_yy(row,col);
            W = [Wxx Wxy; Wxy Wyy];
            E = eig(W);
            E = sort(E);
            min_eig = E(1);
            eig_min(row,col) = min_eig;
        end
    end
    
    for row=1:13:wrows-13
        for col=1:13:wcols-13
            mosaic = eig_min(row:row+12,col:col+12);
            mosaic_row = reshape(mosaic,1,[]);
            max_eig = max(mosaic_row);
            mosaic(find(mosaic<max_eig)) = 0;
            eig_min(row:row+12,col:col+12) = mosaic;
        end
    end
    
    row_eig_min = reshape(eig_min, 1, []);
    sorted_eig_min = sort(row_eig_min,'descend');
    threshold = sorted_eig_min(200);
    [index_i, index_j] = find(eig_min>=threshold);
    
    rgb_pic = cat(3,pic,pic,pic);
    
    figH = figure;
    imshow(uint8(rgb_pic));
    hold on;
    
    for r=1:200
        y = index_i(r)-7;
        x = index_j(r)-7;
        rectangle('Position', [x y 3 3], 'edgecolor', 'r')
    end
    
    baseName = FNames{p}(1:find(FNames{p}=='.')-1);
    figName = strcat(baseName,'_corner_detector_results.jpg');
    
    print(figH,'-djpeg',figName);
    
    % the double summation is taken care of when we do the convolution with
    % 13 by 13 gkern matrix
end