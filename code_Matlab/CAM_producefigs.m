close all; clear; clc 
cd_name = 'D:\xiner\Research2019\MachineLearning_ClassificationOneEar_2021Oct\CAM_Data\all_examples\CAM_heatmaps';
cd(cd_name);
%% %% %% part2, calculate the average of all the heatmaps and plot figs
load('heatAll_0.mat')
heat0 = mean(ht,1);
h_0 = reshape(heat0,32,23);
h0 = (h_0-min(min(h_0)))/(max(max(h_0))-min(min(h_0))); % normalize
load('heatAll_1.mat')
heat1 = mean(ht,1);
h_1 = reshape(heat1,32,23);
h1 = (h_1-min(min(h_1)))/(max(max(h_1))-min(min(h_1)));
load('heatAll_2.mat')
heat2 = mean(ht,1);
h_2 = reshape(heat2,32,23);
h2 = (h_2-min(min(h_2)))/(max(max(h_2))-min(min(h_2)));
load('heatAll_3.mat')
heat3 = mean(ht,1);
h_3 = reshape(heat3,32,23);
h3 = (h_3-min(min(h_3)))/(max(max(h_3))-min(min(h_3)));
load('heatAll_4.mat')
heat4 = mean(ht,1);
h_4 = reshape(heat4,32,23);
h4 = (h_4-min(min(h_4)))/(max(max(h_4))-min(min(h_4)));
load('heatAll_5.mat')
heat5 = mean(ht,1);
h_5 = reshape(heat5,32,23);
h5 = (h_5-min(min(h_5)))/(max(max(h_5))-min(min(h_5)));
load('heatAll_6.mat')
heat6 = mean(ht,1);
h_6 = reshape(heat6,32,23);
h6 = (h_6-min(min(h_6)))/(max(max(h_6))-min(min(h_6)));
load('heatAll_7.mat')
heat7 = mean(ht,1);
h_7 = reshape(heat7,32,23);
h7 = (h_7-min(min(h_7)))/(max(max(h_7))-min(min(h_7)));
load('heatAll_8.mat')
heat8 = mean(ht,1);
h_8 = reshape(heat8,32,23);
h8 = (h_8-min(min(h_8)))/(max(max(h_8))-min(min(h_8)));

figure(1)
imagesc([0 20],[27 43],flipud(h0)); hold on   %%flipud()-rotate
colormap(flipud(gray));
set(gca,'FontName','Times New Roman','FontSize',20,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[])
% print heatmapall_1.eps -depsc2 -r600
figure(2)
imagesc([0 20],[27 43],flipud(h1)); hold on   
colormap(flipud(gray));
set(gca,'FontName','Times New Roman','FontSize',20,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[])
% print heatmapall_2.eps -depsc2 -r600
figure(3)
imagesc([0 20],[27 43],flipud(h2)); hold on   
colormap(flipud(gray));
set(gca,'FontName','Times New Roman','FontSize',20,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[])
% print heatmapall_3.eps -depsc2 -r600
figure(4)
imagesc([0 20],[27 43],flipud(h3)); hold on   
colormap(flipud(gray));
set(gca,'FontName','Times New Roman','FontSize',20,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[])
% print heatmapall_4.eps -depsc2 -r600
figure(5)
imagesc([0 20],[27 43],flipud(h4)); hold on   
colormap(flipud(gray));
set(gca,'FontName','Times New Roman','FontSize',20,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[])
% print heatmapall_5.eps -depsc2 -r600
figure(6)
imagesc([0 20],[27 43],flipud(h5)); hold on   
colormap(flipud(gray));
set(gca,'FontName','Times New Roman','FontSize',20,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[])
% print heatmapall_6.eps -depsc2 -r600
% colorbar
% axis off
% print colorbar_flipudnolabel.eps -depsc2 -r600
figure(7)
imagesc([0 20],[27 43],flipud(h6)); hold on   
colormap(flipud(gray));
set(gca,'FontName','Times New Roman','FontSize',20,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[])
% print heatmapall_7.eps -depsc2 -r600
figure(8)
imagesc([0 20],[27 43],flipud(h7)); hold on   
colormap(flipud(gray));
set(gca,'FontName','Times New Roman','FontSize',20,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[])
% print heatmapall_8.eps -depsc2 -r600
figure(9)
imagesc([0 20],[27 43],flipud(h8)); hold on   
colormap(flipud(gray));
set(gca,'FontName','Times New Roman','FontSize',20,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[])
% print heatmapall_9.eps -depsc2 -r600

%% Subplot all%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(10)
subplot(331)
imagesc([0 20],[27 43],flipud(h0)); hold on   %%flipud()-rotate
colormap(flipud(gray))
colormap('gray');
title('Close-No')
set(gca,'FontName','Times New Roman','FontSize',16,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[40 35 30])
subplot(332)
imagesc([0 20],[27 43],flipud(h1)); hold on   
colormap(flipud(gray))
title('Close-Close')
set(gca,'FontName','Times New Roman','FontSize',16,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[])
subplot(333)
imagesc([0 20],[27 43],flipud(h2)); hold on   
colormap(flipud(gray))
title('Close-Open')
set(gca,'FontName','Times New Roman','FontSize',16,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[])
subplot(334)
imagesc([0 20],[27 43],flipud(h3)); hold on   
colormap(flipud(gray))
title('Open-No')
set(gca,'FontName','Times New Roman','FontSize',16,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[40 35 30])
subplot(335)
imagesc([0 20],[27 43],flipud(h4)); hold on   
colormap(flipud(gray))
title('Open-Close')
set(gca,'FontName','Times New Roman','FontSize',16,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[])
subplot(336)
imagesc([0 20],[27 43],flipud(h5)); hold on   
colormap(flipud(gray))
title('Open-Open')
% colorbar
set(gca,'FontName','Times New Roman','FontSize',16,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[],'YTickLabel',[])
subplot(337)
imagesc([0 20],[27 43],flipud(h6)); hold on   
colormap(flipud(gray))
title('No-No')
set(gca,'FontName','Times New Roman','FontSize',16,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[0 10 20],'YTickLabel',[40 35 30])
subplot(338)
imagesc([0 20],[27 43],flipud(h7)); hold on   
colormap(flipud(gray))
title('No-Close')
set(gca,'FontName','Times New Roman','FontSize',16,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[0 10 20],'YTickLabel',[])
subplot(339)
imagesc([0 20],[27 43],flipud(h8)); hold on   
colormap(flipud(gray))
title('No-Open')
set(gca,'FontName','Times New Roman','FontSize',16,'XTick',[0:10:20],'YTick',[30 35 40],'XTickLabel',[0 10 20],'YTickLabel',[])
% % print colorbar_nrm.eps -depsc2 -r600

% %% part1. print figs to make videos
% load('ht_0.mat')
% heat = [];
% for i = 1:length(ht)    
% heat = reshape(ht(i,:,:),32,23);
% figure(i);
% imagesc(flipud(heat));   %%flipud()-rotate
% colormap('jet');
% colorbar
% axis off
% title(strcat('CAM', num2str(i),': No-Open'))
% a = sprintf('9_%03d',i);
% print(gcf,'-djpeg',strcat(a,'.jpg'))
% end



%% test
% figure(1)
% xname = {'0','','','','','','','','','','','10','','','','','','','','','','','20'};
% yname = {'','','','','40','','','','','','','','','','35','','','','','','','','','','','30','','','','','',''};
% h=heatmap(flipud(h0));
% colormap(flipud(gray))
% h=gca
% YourYticklabel=cell(size(h.YDisplayLabels))
% [YourYticklabel{:}]=deal('');
% h.YDisplayLabels=YourYticklabel














% %% part3, calculate the average of all the Spectrograms and plot figs
% load('ex_0.mat')
% heat0 = mean(ex,1);
% h_0 = reshape(heat0,32,23);
% h0 = (h_0-min(min(h_0)))/(max(max(h_0))-min(min(h_0))); % normalize
% 
% load('ex_1.mat')
% heat1 = mean(ex,1);
% h_1 = reshape(heat1,32,23);
% h1 = (h_1-min(min(h_1)))/(max(max(h_1))-min(min(h_1)));
% 
% load('ex_2.mat')
% heat2 = mean(ex,1);
% h_2 = reshape(heat2,32,23);
% h2 = (h_2-min(min(h_2)))/(max(max(h_2))-min(min(h_2)));
% 
% load('ex_3.mat')
% heat3 = mean(ex,1);
% h_3 = reshape(heat3,32,23);
% h3 = (h_3-min(min(h_3)))/(max(max(h_3))-min(min(h_3)));
% 
% load('ex_4.mat')
% heat4 = mean(ex,1);
% h_4 = reshape(heat4,32,23);
% h4 = (h_4-min(min(h_4)))/(max(max(h_4))-min(min(h_4)));
% 
% load('ex_5.mat')
% heat5 = mean(ex,1);
% h_5 = reshape(heat5,32,23);
% h5 = (h_5-min(min(h_5)))/(max(max(h_5))-min(min(h_5)));
% 
% load('ex_6.mat')
% heat6 = mean(ex,1);
% h_6 = reshape(heat6,32,23);
% h6 = (h_6-min(min(h_6)))/(max(max(h_6))-min(min(h_6)));
% 
% load('ex_7.mat')
% heat7 = mean(ex,1);
% h_7 = reshape(heat7,32,23);
% h7 = (h_7-min(min(h_7)))/(max(max(h_7))-min(min(h_7)));
% 
% load('ex_8.mat')
% heat8 = mean(ex,1);
% h_8 = reshape(heat8,32,23);
% h8 = (h_8-min(min(h_8)))/(max(max(h_8))-min(min(h_8)));
% 
% figure;
% subplot(331)
% imagesc(flipud(h0)); hold on   %%flipud()-rotate
% colormap('jet');
% axis off
% title('S1: Close-No')
% set(gca, 'fontsize',16)
% colorbar
% 
% subplot(332)
% imagesc(flipud(h1)); hold on   
% colormap('jet');
% axis off
% title('S2: Close-Close')
% set(gca, 'fontsize',16)
% colorbar
% 
% subplot(333)
% imagesc(flipud(h2)); hold on   
% colormap('jet');
% axis off
% title('S3: Close-Open')
% set(gca, 'fontsize',16)
% colorbar
% 
% subplot(334)
% imagesc(flipud(h3)); hold on   
% colormap('jet');
% axis off
% title('S4: Open-No')
% set(gca, 'fontsize',16)
% colorbar
% 
% subplot(335)
% imagesc(flipud(h4)); hold on   
% colormap('jet');
% axis off
% title('S5: Open-Close')
% set(gca, 'fontsize',16)
% colorbar
% 
% subplot(336)
% imagesc(flipud(h5)); hold on   
% colormap('jet');
% axis off
% title('S6: Open-Open')
% set(gca, 'fontsize',16)
% colorbar
% 
% subplot(337)
% imagesc(flipud(h6)); hold on   
% colormap('jet');
% axis off
% title('S7: No-No')
% set(gca, 'fontsize',16)
% colorbar
% 
% subplot(338)
% imagesc(flipud(h7)); hold on   
% colormap('jet');
% axis off
% title('S8: No-Close')
% set(gca, 'fontsize',16)
% colorbar
% 
% subplot(339)
% imagesc(flipud(h8)); hold on   
% colormap('jet');
% axis off
% colorbar
% title('S9: No-Open')
% set(gca, 'fontsize',16)
% colorbar

