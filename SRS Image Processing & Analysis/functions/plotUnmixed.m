function plotUnmixed(im, axis)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if exist('axis') == 1
    figure;
    ax1 = axes;
    imagesc(ax1, max(im(:,:,2),0))
    caxis([0 axis(1)])
    colormap(copper)
    freezeColors
    ax2 = axes;
    imagesc(ax2, flipud(max(im(:,:,1),0)));
    h = pcolor(flipud(max(im(:,:,1),0)));
    set(h, 'facealpha', .4);
    shading interp; 
    colormap(jet);
    caxis([0 axis(2)])
    % set(gcf,'color', 'white')
    linkaxes([ax1,ax2])
    ax2.Visible = 'off';
    ax2.XTick = [];
    ax2.YTick = [];
    ax1.XTick = [];
    ax1.YTick = [];
    unfreezeColors;
    colormap(ax1,'copper');
    colormap(ax2,'jet');
    set([ax1,ax2],'Position',[.17 .11 .685 .815]);
    unfreezeColors;
else
    figure;
    ax1 = axes;
    imagesc(ax1, max(im(:,:,2),0))
    colormap(copper)
    freezeColors
    ax2 = axes;
    imagesc(ax2, flipud(max(im(:,:,1),0)));
    h = pcolor(flipud(max(im(:,:,1),0)));
    set(h, 'facealpha', .4);
    shading interp; 
    colormap(jet);
    % set(gcf,'color', 'white')
    linkaxes([ax1,ax2])
    ax2.Visible = 'off';
    ax2.XTick = [];
    ax2.YTick = [];
    ax1.XTick = [];
    ax1.YTick = [];
    unfreezeColors;
    colormap(ax1,'copper');
    colormap(ax2,'jet');
    unfreezeColors;
end
end

