% function punition
% This function allow to detect if there is fake detection event near the
% best detected event (a 20frame windows is opened)
% For each fake event a penalty of 30 frame is added

function[diff_final] = calcul_penalty(event_detected,ref_event)
    [diff_final,pos] = min(abs(ref_event-event_detected));
for ind_indice =1:size(event_detected,2)
    if ind_indice == pos
    elseif abs(ref_event-event_detected(ind_indice))<20
        % Disadvantages for each fake event near the best case
        diff_final = diff_final+30;
    end
end
end