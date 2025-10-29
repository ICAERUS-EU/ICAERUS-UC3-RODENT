clear;clc;
close all; 
set(0, 'defaultfigurecolor', [1 1 1]);

[kalibracija,fs]=audioread("kalibracija ICAERUS.wav");
RMS_kalibr=sqrt(1/length(kalibracija)*sum(kalibracija.^2));
L_kalibr=20*log10(RMS_kalibr/(2*10^-5));
faktor=94.4-L_kalibr;
faktor_t=10^(faktor/20);

korak=1;
theta=1:korak:30;
p0=2*10^(-5);

ap=0.1;
as=80;
f=[20000; 30000; 40000; 50000; 60000];
wp=[f-1000 f+1000] * 2 * pi / fs;
ws=[f-1500 f+1500] * 2 * pi / fs;

for i=1:length(f)
    apl=min(1-10^(-ap/20),10^(ap/20)-1);
    asl=10^(-as/20);
    [n,ff,a,wt]=firpmord([ws(i,1) wp(i,1) wp(i,2) ws(i,2)]/pi,[0 1 0],[asl apl asl]);
    h(i,:)=firpm(n,ff,a,wt);
    [H(i,:),w]=freqz(h(i,:),1,fs);
    
end


for br=1:length(theta)
    theta(br)
    load(strcat("signali\elevacija_",num2str(theta(br)),".mat"));
    x=x*faktor_t;
    for i=1:length(f)
        y=filter(h(i,:),1,x);
        eff_value(i,br+1,:)=rms(y);   
    end
end

load("signali\elevacija_0.mat");
x=x*faktor_t;
for i=1:length(f)
    y=filter(h(i,:),1,x);
    eff_value(i,1,:)=ones(1,360)*rms(y);   
end

X=fft(x);
f_osa=(0:length(x(:,1))-1)*fs/length(x(:,1));
figure;plot(f_osa(1:length(X)/2),20*log10(abs(X(1:length(X)/2))./p0));
xlabel('Frequeny [Hz]'); ylabel('Sound Pressure Levele [dB]');
hold on;

eff_value(:,:,361)=eff_value(:,:,1);  %% da bi popunjen bio krug, iako je 0 i 360 stepeni isto

%%%% prikaz nivoa na 1m odstojanja
t = (0:korak:360)*pi/180;          % Azimuth angle
p = (0:korak:30)*pi/180;           % Polar angle
[Theta_kapa, Phi_kapa] = meshgrid(t, p);
X_dole = tan(Phi_kapa) .* cos(Theta_kapa);
Y_dole = tan(Phi_kapa) .* sin(Theta_kapa);
Z_dole = 0*ones(size(X_dole));

for i=1:length(f)
    L_niz_osn=squeeze(eff_value(i,:,:));
    L_niz=20*log10(abs(L_niz_osn./p0));
    % L_niz=20*log10(abs(L_niz_osn./max(max(abs(L_niz_osn)))));
    normalize(i)=L_niz(1,1);
    ax=figure,surf(X_dole, Y_dole, Z_dole, L_niz, 'EdgeColor', 'none');
    ax = gca; 
    % ax.CLim = [-40 0];
     ax.CLim = [35 95]; %% SPL
    col=colorbar; 
    ylabel(col,'Level [dB]','FontSize',12);
    xlabel('X [m]','FontSize',12),ylabel('Y [m]','FontSize',12),
    xticks(-0.6:0.2:0.6);
    yticks(-0.6:0.2:0.6);ylim([-0.605 0.605])
    title(strcat('\rm f=',num2str(f(i))," Hz"));
    view(2);
    axis equal;

    % exportgraphics(ax,strcat('slike\merenje_norm_',num2str(f(i)),'Hz.emf'),'Resolution',300);

    J_std(i)=std(reshape(L_niz,1,[]));

end