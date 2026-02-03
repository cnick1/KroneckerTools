function status = odeprog(t, y, flag)
%odeprog Command window progress bar for ODE solvers
% 
%   Usage (example): 
%       global T0; opts = odeset(OutputFcn=@odeprog);
%       T0 = tic; [T, Z] = ode45(@(t, z) F(z), [0, 10], x0, opts);
%
%%
persistent T1 nSteps lastPct lastUpdateTime
global T0

status = false;

switch flag
    case 'init'
        % Initialize progress bar
        elapsed = toc(T0);
        T1 = t(end);
        nSteps = 50; % Number of blocks in the progress bar
        lastPct = -1;
        lastUpdateTime = 0;
        fprintf(' |%s|  (elapsed: %5i s, remaining: ----- s)', repmat(' ',1,nSteps), round(elapsed));
        
    case ''
        % ODE solver step
        if isempty(t), return; end
        tNow = t(end);
        pct = min(100, max(0, 100 * tNow / T1));
        block = floor(pct / (100/nSteps));
        elapsed = toc(T0);
        eta = (elapsed / max(tNow,eps)) * (T1 - tNow); % avoid divide-by-zero
        needsUpdate = pct - lastPct >= 2 || block == nSteps;
        timeSinceLast = elapsed - lastUpdateTime;
        
        if needsUpdate || timeSinceLast >= 1
            bar = [repmat('-',1,block), repmat(' ',1,nSteps-block)];
            fprintf(repmat('\b',1,93));
            fprintf(' |%s|  (elapsed: %5i s, remaining: %5i s)', bar, round(elapsed), min(round(eta),99999));
            if needsUpdate
                lastPct = pct;
            end
            lastUpdateTime = elapsed;
        end
        
    case 'done'
        % Finalize
        % elapsed = toc(T0);
        % bar = repmat('-',1,nSteps);
        fprintf(repmat('\b',1,93));
        % fprintf(' |%s|  (elapsed: %5i s, remaining:     0 s)\n', bar, round(elapsed));
        clear T0 T1 nSteps lastPct lastUpdateTime
end
end