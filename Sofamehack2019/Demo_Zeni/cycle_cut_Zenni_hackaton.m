% FONCTION cycles_cut
% Allow to detrmine gait event from a c3d file

% Input: H : acqusition file extracted from btk

% Output : HS a list of the heel strike event
%          TOFF a list of the too off event

% Article:
%"Two simple methods for determining gait events during treadmilland
%overground walking using kinematic data"
% Volume 27, Issue 4, May 2008, Pages 710–714
% J.A. Zeni Jr.a, , , J.G.Richardsb, J.S. Higginsonc


function [HS,TOFF] = cycle_cut_Zenni_hackaton(H,side)
f = btkGetPointFrequency(H); % Fréquence d'acquisition
FF = btkGetFirstFrame(H); % first frame
point = btkGetMarkers(H); % Position de tous les marqueurs au cours du temps

% Side choice
if strcmpi(side,'left')
    point.HEE = point.LHEE;
    point.TOE = point.LTOE;
elseif strcmpi(side,'right')
    point.HEE = point.RHEE;
    point.TOE = point.RTOE;
end


%% Determination of the first valid frame
if strcmpi(side,'left')
    markerList = {'LASI','LPSI','RASI','RPSI','LTOE','LHEE'};
elseif strcmpi(side,'right')
    markerList =  {'LASI','LPSI','RASI','RPSI','RTOE','RHEE'};
end

maskfinal = and(point.(markerList{1}),point.(markerList{2}));

for i = 3:size(markerList,2)
    maskfinal = and(maskfinal,point.(markerList{i}));
end

[~,ideb] = max(maskfinal);
[~,ifin] = max(maskfinal(end:-1:1));

ideb = ideb(1) + 10;
ifin = size(maskfinal,1)-ifin;
ifin = ifin-10;


%% Data filtering
fc = 10;
% Filtre Butterworth 4eme order de frequence de coupure fc
[B,A] = butter(4,fc./(f/2),'low');

markerList = {'LASI','RASI','LPSI','RPSI',...
    'TOE','HEE'};

for markerPosition = 1:size(markerList,2)
    %Initialisation
    init = point.(markerList{markerPosition})(ideb:ifin,:);
    initfilt = zeros(size(init));
    n = size(init,1);
    for collumn = 1 : 3
        % interpolation
        x = 1:n;
        y = init(:,collumn);
        xx = 1:1:n;
        temp = interp1(x,y,xx,'spline');
        %filtre
        initfilt(:,collumn) = filtfilt(B,A,temp);
    end
    data.(markerList{markerPosition}) = initfilt;
end


%% calcul du repère Bassin.

OrPelvis = ((data.LPSI+data.RPSI)/2+(data.LASI+data.RASI)/2)/2;
xPelvis = (data.LASI+data.RASI)/2-(data.LPSI+data.RPSI)/2;

% We remove the vertical component of the pelvis to have only the walking
% direction (we remove the effect of the pelvis tilting)
xPelvis(:,3) = zeros(size(xPelvis,1),1);

% Position of the heel and toe in the pelvis frame (only antero posterior)
% for detecting the foot strike we use the heel marker
tHS = dot(xPelvis,data.HEE-OrPelvis,2);
% for detecting the foot off we use the toe marker
tTO = dot(xPelvis,data.TOE-OrPelvis,2);

% We find the maximal value for tHs and the minimal value for tTO(so we use
% -tTO for findpeaks
[~,HS] = findpeaks(tHS);
[~,TO] = findpeaks(-tTO);
% We analysed only for the valid frame so we have to put it back in the inital 
% time frame of the file
HS = HS + FF - 1 + ideb; % heel strike
TOFF = TO + FF - 1+ ideb; % Toe off
end

