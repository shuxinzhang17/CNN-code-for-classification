close all; clear all; clc
cd_name = 'D:\xiner\Research2019\MachineLearning_ClassificationOneEar_2021Oct\mul_validation';
cd(cd_name)
%% multiple validation average accuracy
%% traning data without dropout
load('dropout1110_val2acc1.mat'); acc1 = val2acc;
load('dropout1110_val2acc2.mat'); acc2 = val2acc;
load('dropout1110_val2acc3.mat'); acc3 = val2acc;
load('dropout1110_val2acc4.mat'); acc4 = val2acc;
load('dropout1110_val2acc5.mat'); acc5 = val2acc;
load('dropout1110_val2acc6.mat'); acc6 = val2acc;
load('dropout1110_val2acc7.mat'); acc7 = val2acc;
load('dropout1110_val2acc8.mat'); acc8 = val2acc;
load('dropout1110_val2acc9.mat'); acc9 = val2acc;
load('dropout1110_val2acc10.mat'); acc10 = val2acc;
load('dropout1110_val2acc11.mat'); acc11 = val2acc;
load('dropout1110_val2acc12.mat'); acc12 = val2acc;
load('dropout1110_val2acc13.mat'); acc13 = val2acc;
load('dropout1110_val2acc14.mat'); acc14 = val2acc;
load('dropout1110_val2acc15.mat'); acc15 = val2acc;
%% validation accuracy without dropout
load('dropout1110_valacc1.mat'); valacc1 = valacc;
load('dropout1110_valacc2.mat'); valacc2 = valacc;
load('dropout1110_valacc3.mat'); valacc3 = valacc;
load('dropout1110_valacc4.mat'); valacc4 = valacc;
load('dropout1110_valacc5.mat'); valacc5 = valacc;
load('dropout1110_valacc6.mat'); valacc6 = valacc;
load('dropout1110_valacc7.mat'); valacc7 = valacc;
load('dropout1110_valacc8.mat'); valacc8 = valacc;
load('dropout1110_valacc9.mat'); valacc9 = valacc;
load('dropout1110_valacc10.mat'); valacc10 = valacc;
load('dropout1110_valacc11.mat'); valacc11 = valacc;
load('dropout1110_valacc12.mat'); valacc12 = valacc;
load('dropout1110_valacc13.mat'); valacc13 = valacc;
load('dropout1110_valacc14.mat'); valacc14 = valacc;
load('dropout1110_valacc15.mat'); valacc15 = valacc;
% figure(1)
% epoch = 1:100;
% plot(epoch,acc1*100,'k-','linewidth',2); hold on
% plot(epoch,acc2*100,'k--','linewidth',2); hold on
% plot(epoch,acc3*100,'k:','linewidth',2); hold on
% plot(epoch,acc4*100,'b-','linewidth',2); hold on
% plot(epoch,acc5*100,'b--','linewidth',2); hold on
% plot(epoch,acc6*100,'b:','linewidth',2); hold on
% plot(epoch,acc7*100,'r-','linewidth',2); hold on
% plot(epoch,acc8*100,'r--','linewidth',2); hold on
% plot(epoch,acc9*100,'r:','linewidth',2); hold on
% plot(epoch,acc10*100,'c-','linewidth',2); hold on
% plot(epoch,acc11*100,'c--','linewidth',2); hold on
% plot(epoch,acc12*100,'c:','linewidth',2); hold on
% plot(epoch,acc13*100,'m-','linewidth',2); hold on
% plot(epoch,acc14*100,'m--','linewidth',2); hold on
% plot(epoch,acc15*100,'m:','linewidth',2); grid on
% legend('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15')
% set(gca,'xtick',[0:20:100],'ytick',[0:20:100]);
% title('training accuracy')
% % legend(['training acc(train data)=',num2str(sprintf('%.2f',acc(end)*100)),'%'],['val-acc(validation data)=',num2str(sprintf('%.2f',valacc(end)*100)),'%'],['val2-acc(train data)=',num2str(sprintf('%.2f',val2acc(end)*100)),'%'])
% set(gca,'FontName','Times New Roman','fontsize',16,'xtick',[0:20:100],'ytick',[0:20:100]);
% % set(gca,'FontName','Times New Roman','fontsize',16,'xtick',[0:20:100],'ytick',[0:20:100],'xticklabel',[0:20:100],'yticklabel',[0:20:100]);


%% combine some data together
Acc = [acc2;acc3;acc4;acc5;acc6;acc9;acc11;acc13;acc14;acc15];
Val_acc = [valacc2;valacc3;valacc4;valacc5;valacc6;valacc9;valacc11;valacc13;valacc14;valacc15];
average_Acc = mean(Acc);
acc_std = std(Acc);
average_Valacc = mean(Val_acc);
valacc_std = std(Val_acc);
figure(2)  %%[102/255,0,0] / [1,102/255,0]
epoch = 1:100;
plot(epoch,average_Valacc*100,'-','color',[.5,.5,.5],'linewidth',2); hold on %[.5,.5,.5]
plot(epoch,(average_Valacc+valacc_std)*100,'-.','color',[.5,.5,.5],'linewidth',0.8); hold on 
plot(epoch,(average_Valacc-valacc_std)*100,'-.','color',[.5,.5,.5],'linewidth',0.8); hold on
plot(epoch,average_Acc*100,'-','color',[0,0,0],'linewidth',2); hold on %'k-'
plot(epoch,(average_Acc+acc_std)*100,'-.','color',[0,0,0],'linewidth',0.8); hold on
plot(epoch,(average_Acc-acc_std)*100,'-.','color',[0,0,0],'linewidth',0.8); hold on
xlim([0 102])
ylim([0 102])
% set(gca,'FontName','Times New Roman','fontsize',18,'xtick',[0:20:100],'ytick',[0:20:100],'xticklabel',[],'yticklabel',[]);
set(gca,'FontName','Times New Roman','fontsize',20,'xtick',[0:20:100],'ytick',[0:20:100],'xticklabel',[0:20:100],'yticklabel',[0:20:100]);
xlabel('Epoch')
ylabel('Accuracy [%]')
% % print acc_valacc6_nolabel.eps -depsc -r600 

figure(3)
epoch = 1:100;
plot(epoch,average_Valacc*100,'-','color',[.5,.5,.5],'linewidth',2.5); hold on
plot(epoch,(average_Valacc+valacc_std)*100,'-.','color',[.5,.5,.5],'linewidth',0.8); hold on
plot(epoch,(average_Valacc-valacc_std)*100,'-.','color',[.5,.5,.5],'linewidth',0.8); hold on
plot(epoch,average_Acc*100,'-','color',[0,0,0],'linewidth',2.5); hold on
plot(epoch,(average_Acc+acc_std)*100,'-.','color',[0,0,0],'linewidth',0.8); hold on
plot(epoch,(average_Acc-acc_std)*100,'-.','color',[0,0,0],'linewidth',0.8); hold on
xlim([80 100])
ylim([90 100])
set(gca,'FontName','Times New Roman','fontsize',24,'xtick',[80:10:100],'ytick',[90:5:100],'xticklabel',[80:10:100],'yticklabel',[90:5:100]);
% % print acc_valacc6_nolabelzoomIn2.eps -depsc -r600 




%% test dropout numbers
% load('dropout1110_acc1.mat')
% load('dropout1110_valacc1.mat')
% load('dropout1110_val2acc1.mat')
% figure(1)
% subplot(351)
% epoch = 1:length(acc);
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% % legend(['training acc(train data)=',num2str(sprintf('%.2f',acc(end)*100)),'%'],['val-acc(validation data)=',num2str(sprintf('%.2f',valacc(end)*100)),'%'],['val2-acc(train data)=',num2str(sprintf('%.2f',val2acc(end)*100)),'%'])
% title('1')
% % set(gca,'FontName','Times New Roman','fontsize',16,'xtick',[0:20:100],'ytick',[0:20:100]);
% subplot(352)
% load('dropout1110_acc2.mat')
% load('dropout1110_valacc2.mat')
% load('dropout1110_val2acc2.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% title('2')
% subplot(353)
% load('dropout1110_acc3.mat')
% load('dropout1110_valacc3.mat')
% load('dropout1110_val2acc3.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% title('3')
% subplot(354)
% load('dropout1110_acc4.mat')
% load('dropout1110_valacc4.mat')
% load('dropout1110_val2acc4.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% title('4')
% subplot(355)
% load('dropout1110_acc5.mat')
% load('dropout1110_valacc5.mat')
% load('dropout1110_val2acc5.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% title('5')
% subplot(356)
% load('dropout1110_acc6.mat')
% load('dropout1110_valacc6.mat')
% load('dropout1110_val2acc6.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% ylabel('Accuracy [%]')
% title('6')
% subplot(357)
% load('dropout1110_acc7.mat')
% load('dropout1110_valacc7.mat')
% load('dropout1110_val2acc7.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% title('7')
% subplot(358)
% load('dropout1110_acc8.mat')
% load('dropout1110_valacc8.mat')
% load('dropout1110_val2acc8.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% title('8')
% subplot(359)
% load('dropout1110_acc9.mat')
% load('dropout1110_valacc9.mat')
% load('dropout1110_val2acc9.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% title('9')
% subplot(3,5,10)
% load('dropout1110_acc10.mat')
% load('dropout1110_valacc10.mat')
% load('dropout1110_val2acc10.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% title('10')
% subplot(3,5,11)
% load('dropout1110_acc11.mat')
% load('dropout1110_valacc11.mat')
% load('dropout1110_val2acc11.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% title('11')
% subplot(3,5,12)
% load('dropout1110_acc12.mat')
% load('dropout1110_valacc12.mat')
% load('dropout1110_val2acc12.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% title('12')
% subplot(3,5,13)
% load('dropout1110_acc13.mat')
% load('dropout1110_valacc13.mat')
% load('dropout1110_val2acc13.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% xlabel('Epoch')
% title('13')
% subplot(3,5,14)
% load('dropout1110_acc14.mat')
% load('dropout1110_valacc14.mat')
% load('dropout1110_val2acc14.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% title('14')
% subplot(3,5,15)
% load('dropout1110_acc15.mat')
% load('dropout1110_valacc15.mat')
% load('dropout1110_val2acc15.mat')
% plot(epoch,acc*100,'r-','linewidth',2); hold on
% plot(epoch,valacc*100,'b-','linewidth',2); hold on
% plot(epoch,val2acc*100,'g-','linewidth',2); grid on
% title('15')

% %% test dropout numbers
% load('acc_Dropout0110.mat')
% load('valacc_Dropout0110.mat')
% figure(1)
% epoch = 1:length(acc);
% plot(epoch,acc*100,'b-','linewidth',2); hold on
% plot(epoch,valacc*100,'r-','linewidth',2);
% grid on
% legend(['training acc=',num2str(sprintf('%.2f',acc(end)*100)),'%'],['val-acc=',num2str(sprintf('%.2f',valacc(end)*100)),'%'])
% xlabel('Epoch')
% ylabel('Accuracy [%]')
% title('No dropout 0110')
% set(gca,'FontName','Times New Roman','fontsize',16,'xtick',[0:20:100],'ytick',[0:20:100]);
% 
% load('acc_Dropout1011.mat')
% load('valacc_Dropout1011.mat')
% figure(2)
% epoch = 1:length(acc);
% plot(epoch,acc*100,'b-','linewidth',2); hold on
% plot(epoch,valacc*100,'r-','linewidth',2);
% grid on
% legend(['training acc=',num2str(sprintf('%.2f',acc(end)*100)),'%'],['val-acc=',num2str(sprintf('%.2f',valacc(end)*100)),'%'])
% xlabel('Epoch')
% ylabel('Accuracy [%]')
% title('No dropout 1011')
% set(gca,'FontName','Times New Roman','fontsize',16,'xtick',[0:20:100],'ytick',[0:20:100]);
% 
% load('acc_Dropout0111.mat')
% load('valacc_Dropout0111.mat')
% figure(3)
% epoch = 1:length(acc);
% plot(epoch,acc*100,'b-','linewidth',2); hold on
% plot(epoch,valacc*100,'r-','linewidth',2);
% grid on
% legend(['training acc=',num2str(sprintf('%.2f',acc(end)*100)),'%'],['val-acc=',num2str(sprintf('%.2f',valacc(end)*100)),'%'])
% xlabel('Epoch')
% ylabel('Accuracy [%]')
% title('No dropout 0111')
% set(gca,'FontName','Times New Roman','fontsize',16,'xtick',[0:20:100],'ytick',[0:20:100]);
% 
% load('acc_Dropout1110.mat')
% load('valacc_Dropout1110.mat')
% figure(4)
% epoch = 1:length(acc);
% plot(epoch,acc*100,'b-','linewidth',2); hold on
% plot(epoch,valacc*100,'r-','linewidth',2);
% grid on
% legend(['training acc=',num2str(sprintf('%.2f',acc(end)*100)),'%'],['val-acc=',num2str(sprintf('%.2f',valacc(end)*100)),'%'])
% xlabel('Epoch')
% ylabel('Accuracy [%]')
% title('No dropout 1110')
% set(gca,'FontName','Times New Roman','fontsize',16,'xtick',[0:20:100],'ytick',[0:20:100]);


%% test dropout in different layers
% load('acc_Dropout0000.mat')
% load('valacc_Dropout0000.mat')
% figure(1)
% epoch = 1:82;
% plot(epoch,acc*100,'b-','linewidth',2); hold on
% plot(epoch,valacc*100,'r-','linewidth',2);
% grid on
% legend(['training acc=',num2str(sprintf('%.2f',acc(end)*100)),'%'],['val-acc=',num2str(sprintf('%.2f',valacc(end)*100)),'%'])
% xlabel('Epoch')
% ylabel('Accuracy [%]')
% title('No dropout 0000')
% set(gca,'FontName','Times New Roman','fontsize',16,'xtick',[0:20:100],'ytick',[0:20:100]);
% 
% load('acc_Dropout0001.mat')
% load('valacc_Dropout0001.mat')
% figure(2)
% epoch = 1:70;
% plot(epoch,acc*100,'b-','linewidth',2); hold on
% plot(epoch,valacc*100,'r-','linewidth',2);
% grid on
% legend(['training acc=',num2str(sprintf('%.2f',acc(end)*100)),'%'],['val-acc=',num2str(sprintf('%.2f',valacc(end)*100)),'%'])
% xlabel('Epoch')
% ylabel('Accuracy [%]')
% title('One dropout 0001')
% xlim([0 70])
% ylim([0 100])
% set(gca,'FontName','Times New Roman','fontsize',16);
% 
% load('acc_Dropout1110_3.mat')
% load('valacc_Dropout1110_3.mat')
% figure(3)
% epoch = 1:100;
% plot(epoch,acc*100,'b-','linewidth',2); hold on
% plot(epoch,valacc*100,'r-','linewidth',2);
% grid on
% legend(['training acc=',num2str(sprintf('%.2f',acc(end)*100)),'%'],['val-acc=',num2str(sprintf('%.2f',valacc(end)*100)),'%'])
% xlabel('Epoch')
% ylabel('Accuracy [%]')
% title('Dropout 1110')
% xlim([0 100])
% ylim([0 100])
% set(gca,'FontName','Times New Roman','fontsize',16);
