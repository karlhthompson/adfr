%% Import data from spreadsheet

tic
if exist('APMR','var')
else
    % Initialize variables.
    filename = 'C:\Users\karlh\Box\Documents\MATLAB\803049246_T_DB1B_MARKET.csv';
    delimiter = ',';
    startRow = 2;
    
    % Format for each line of text:
    formatSpec = '%q%q%f%f%f%[^\n\r]';
    
    % Open the text file.
    fileID = fopen(filename,'r');
    
    % Read columns of data according to the format.
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, ...
        'TextType', 'string', 'EmptyValue', NaN, ...
        'HeaderLines' , startRow-1, 'ReturnOnError', ...
        false, 'EndOfLine', '\r\n');
    
    % Close the text file.
    fclose(fileID);
    
    % Create output variable.
    APMR_tab = table(dataArray{1:end-1}, 'VariableNames', {'ORIGIN', ...
        'DEST','PASSENGERS','MARKET_FARE','MARKET_DISTANCE'});
    
    % Remove duplicate rows.
    APMR_tab = grpstats(APMR_tab,{'ORIGIN','DEST'});
    
    % Transform table into array.
    APMR = table2array(APMR_tab);
    total_num_pass = sum(round(str2double(APMR(:,3)).*str2double(APMR(:,4))));
    %APMR(29:end,:) = []; % Choose how much data to include
    
    % Clear temporary variables.
    clearvars filename delimiter startRow formatSpec fileID dataArray ans APMR_tab;
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%% Test Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TEST_ARRAY = ["AAA"; "BBB"; "BBB"; "CCC"; "CCC"; "DDD"; "AAA"; "EEE"; ...
%     "AAA"; "FFF"; "BBB"; "FFF"; "CCC"; "GGG"; "DDD"; "GGG"; ...
%     "DDD"; "HHH"; "EEE"; "FFF"; "FFF"; "GGG"; "GGG"; "HHH"; ...
%     "EEE"; "III"; "FFF"; "JJJ"; "GGG"; "KKK"; "HHH"; "LLL"; ...
%     "III"; "JJJ"; "JJJ"; "KKK"; "KKK"; "LLL"; "III"; "MMM"; ...
%     "JJJ"; "MMM"; "JJJ"; "NNN"; "KKK"; "OOO"; "KKK"; "PPP"; ...
%     "LLL"; "PPP"; "MMM"; "NNN"; "NNN"; "OOO"; "OOO"; "PPP"];
% NUMBERS = 0:0.5:27;
% for i=1:2:length(TEST_ARRAY)-1
%     APMR(i-NUMBERS(i),1)=TEST_ARRAY(i);
%     APMR(i-NUMBERS(i),2)=TEST_ARRAY(i+1);
% end
% clearvars TEST_ARRAY NUMBERS i;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Organize imported data

% Defining departure and arrival airports and number of flights
all_departure_airports = categorical(APMR(:,1));
all_arrival_airports = categorical(APMR(:,2));
all_passenger_numbers = round(str2double(APMR(:,3)).*str2double(APMR(:,4)));
all_market_fares = str2double(APMR(:,5));

% Find number of departure airports
dep_airport_unique_list = unique(categorical(APMR(:,1)));
dep_air_num = hist(categorical(APMR(:,1)),dep_airport_unique_list)';
dep_airport_count = length(dep_air_num);

% Find number of arrival airports
arr_airport_unique_list = unique(categorical(APMR(:,2)));
arr_air_num = hist(categorical(APMR(:,2)),arr_airport_unique_list)';
arr_airport_count = length(arr_air_num);

% Find total number of airports
airport_unique_list = unique(categorical(vertcat(APMR(:,1), APMR(:,2))));
airport_num = hist(categorical(vertcat(APMR(:,1), APMR(:,2))),airport_unique_list)';
airport_count = length(airport_num);

% Replace the airport name values in APMR by their indexes
APMR_num = APMR;
for n2=1:airport_count
    a1 = airport_unique_list(n2);
    a2 = find(categorical(APMR)==a1);
    APMR_num(a2)=n2;
end
APMR_num = str2double(APMR_num);
clearvars a1 a2 n2;

% Remove the outliers in ticket price data
out_cost = APMR_num(:,5);
out_index = out_cost>1200;
out_cost(out_index) = 0;
APMR_num(:,5) = out_cost;
clearvars out_cost out_index

% Nominal flow on each route
Nominal_Y = zeros(airport_count,airport_count);
for n=1:length(APMR_num)
    if APMR_num(n,1) <= APMR_num(n,2)
        Nominal_Y(APMR_num(n,1),APMR_num(n,2)) = ...
            Nominal_Y(APMR_num(n,1),APMR_num(n,2)) + ...
            round(APMR_num(n,3)*APMR_num(n,4));
    else
        Nominal_Y(APMR_num(n,2),APMR_num(n,1)) = ...
            Nominal_Y(APMR_num(n,2),APMR_num(n,1)) + ...
            round(APMR_num(n,3)*APMR_num(n,4));
    end
end
Nominal_Y(Nominal_Y<5) = 0;
Nominal_Y_sym = Nominal_Y + Nominal_Y';

%% Set up the COMBINED ATTACKER-DEFENDER optimization problem

% Define the network state
x_ij = ones(airport_count,airport_count,1);
for n3=1:length(APMR_num)
    x_ij(APMR_num(n3,1),APMR_num(n3,2)) = 0;
end
clearvars n3;

% Define the per-passenger cost of traversing an arc
c_ij = zeros(airport_count,airport_count);
for n=1:length(APMR_num)
    if c_ij(APMR_num(n,1),APMR_num(n,2)) == 0
        c_ij(APMR_num(n,1),APMR_num(n,2)) = (APMR_num(n,5));
    else
        c_ij(APMR_num(n,1),APMR_num(n,2)) = ...
            (c_ij(APMR_num(n,1),APMR_num(n,2)) + (APMR_num(n,5)))/2;
    end
end
c_ij = c_ij + c_ij';
c_ij(c_ij<=0) = 999999; %can be any large enough number

% Define the per-unit penalty cost of traversing an arc if damaged
q_ij = ones(airport_count,airport_count,1);

% Define the cost to 'break' each route
r_ij = ones(airport_count,airport_count,1);
r_ij(Nominal_Y==0) = 99; % Ensuring the attacker doesn't attack non-existent routes

% Define the available attack budget
attack_budget = 5;

% Define the cost to protect existing routes
h_ij = ones(airport_count,airport_count,1);
h_ij(Nominal_Y==0) = 99; % Ensuring the defender doesn't defend non-existent routes

% Define the available defence budget
defence_budget = 5;

% Plot flight distribution map
figure(1)
image(Nominal_Y,'CDataMapping','scaled')
colormap(flipud(gray))
title('Initial Network Traffic')
colorbar

% Define AD Model solution variables
% State of network after attack (1: damaged, 0: intact)
X = optimvar('X',airport_count,airport_count,1,'Type','integer','LowerBound',0,'UpperBound',1);
% Did the defender protect the edge? (1: protected, 0: otherwise)
W = optimvar('W',airport_count,airport_count,1,'Type','integer','LowerBound',0,'UpperBound',1);
% Additional product optimization variable
XW = optimvar('XW',airport_count,airport_count,1,'Type','integer','LowerBound',0,'UpperBound',1);

% Define the AD Model optimization problem
AD_prob = optimproblem('ObjectiveSense','minimize');

% Increase c for high-taffic routes
c_attdef = c_ij.*log10(Nominal_Y_sym);
c_attdef(isinf(c_attdef)) = 999999;
c_attdef(isnan(c_attdef)) = 999999;
c_attdef(c_attdef <= 0) = 999999;

% Calculate the AD Model optimization objective
sumAD1 = sum(sum(((c_attdef + q_ij .* (1 - X - W + XW)) .* Nominal_Y) + ...
    ((c_attdef' + q_ij' .* (1 - X - W + XW)) .* Nominal_Y)));
% Define the AD Model optimization objective
AD_prob.Objective = sumAD1;

% Prepare the optimization constraints
% Attack budget constraint
att_budget_constr = sum(sum(r_ij.*X)) <= attack_budget;
% Defence budget constraint
def_budget_constr = sum(sum(h_ij.*W)) <= defence_budget;
% Initialize product constraints
product_constr1 = optimconstr([airport_count,airport_count]);
product_constr2 = optimconstr([airport_count,airport_count]);
product_constr3 = optimconstr([airport_count,airport_count]);
% Write product Constraints
product_constr1 = XW(:,:) <= X(:,:);
product_constr2 = XW(:,:) <= W(:,:);
product_constr3 = W(:,:) - XW(:,:) <= 1 - X(:,:);

% Define the optimization constraints
AD_prob.Constraints.consAD1 = att_budget_constr;
AD_prob.Constraints.consAD2 = def_budget_constr;
AD_prob.Constraints.consAD3 = product_constr1;
AD_prob.Constraints.consAD4 = product_constr2;
AD_prob.Constraints.consAD5 = product_constr3;

% Performe the optimization using a mixed-integer linear programming algorithm
AD_options = optimoptions('intlinprog');
[WXsol,WXfval,WXexitflag,WXoutput] = solve(AD_prob,AD_options);
Solved_X = WXsol.X(:,:);
Solved_W = WXsol.W(:,:);

% Print out the most vulnerable routes
[att_col, att_row] = find(Solved_X);
vulnerable_routes_dep = strings([length(find(Solved_X)),1]);
vulnerable_routes_arr = strings([length(find(Solved_X)),1]);
for n=1:length(find(Solved_X))
    vulnerable_routes_dep(n) = char(airport_unique_list(att_col(n)));
    vulnerable_routes_arr(n) = char(airport_unique_list(att_row(n)));
end
vulnerable_routes = [vulnerable_routes_dep vulnerable_routes_arr];
fprintf('The most vulnerable routes for an attack budget of %d are:\n', attack_budget)
disp(vulnerable_routes)

% Plot attacked routes map
figure(2)
image(Solved_X,'CDataMapping','scaled')
colormap(flipud(gray))
title('Attacked Routes')

% Print out the protected routes
if sum(sum(Solved_W)) ~= 0
    [def_col, def_row] = find(Solved_W);
    protected_routes_dep = strings([length(find(Solved_W)),1]);
    protected_routes_arr = strings([length(find(Solved_W)),1]);
    for n=1:length(find(Solved_W))
        protected_routes_dep(n) = char(airport_unique_list(def_col(n)));
        protected_routes_arr(n) = char(airport_unique_list(def_row(n)));
    end
    protected_routes = [protected_routes_dep protected_routes_arr];
    fprintf('The routes chosen for protection based on a defence budget of %d are:\n', defence_budget)
    disp(protected_routes)
end

% Plot reinforced routes map
figure(5)
image(Solved_W,'CDataMapping','scaled')
colormap(flipud(gray))
title('Reinforced Routes')

% % Plot the initial network state
% C = convertStringsToChars(APMR(:,1));
% D = convertStringsToChars(APMR(:,2));
% F = str2double(APMR(:,3));
% G = convertStringsToChars(unique(vertcat(APMR(:,1), APMR(:,2))));
% E = digraph(C,D,F,G);
% figure(3)
% plot(E,'Layout','force','EdgeLabel',E.Edges.Weight)
% title('Initial Network State')
% clearvars C D E F G;
%
% % Plot the updated network state
% APMR2 = APMR;
% for n=1:length(att_col)
%     for k=1:length(APMR_num)
%         if APMR_num(k,1:2)==[att_col(n) att_row(n)]
%             APMR2(k,:) = 0;
%         end
%     end
% end
% A = APMR2(:,1); A(A=="0")=[];
% B = APMR2(:,2); B(B=="0")=[];
% C = convertStringsToChars(A);
% D = convertStringsToChars(B);
% E = digraph(C,D);
% figure(4)
% plot(E,'Layout','force')
% title('Updated Network State')
% clearvars A B C D APMR2;

%% Setting up the Flight Rerouting optimization problem

% Update X solution variable
X_updated = round(Solved_X) + x_ij;

% Update the per-unit cost of traversing newly broken routes
c_ij(round(Solved_X)==1) = 999999;

% Define the upper bound on total rerouted undirected flow on an edge
u_ij = 0.5 .* Nominal_Y_sym;

% Define the per-unit penalty cost for demand shortfall
p_n = ones(1,airport_count,1) .* 99999;

% Define solution variables
% Rerouted flow through existing network routes
numArrays = length(unique(vulnerable_routes_dep));
Y = optimvar('Y',airport_count,airport_count,numArrays,'Type','integer','LowerBound',0);
% Flight shortfall at each airport
S = optimvar('S',1,airport_count,numArrays,'Type','integer','LowerBound',0);

% Define the optimization problem
RR_prob = optimproblem('ObjectiveSense','minimize');

% Calculate the Flight Rerouting optimization objective
sumRR1 = 0;
c_rerouting = c_ij./log10(Nominal_Y_sym);
c_rerouting(isinf(c_rerouting)) = 999999;
c_rerouting(isnan(c_rerouting)) = 999999;
c_rerouting(c_rerouting <= 0) = 999999;
for k=1:numArrays
    sumRR1 = sumRR1 + sum(sum(((c_rerouting + q_ij.*X_updated).*Y(:,:,k)) + ...
        ((c_rerouting' + q_ij'.*X_updated).*Y(:,:,k)')));
end

sumRR2 = 0;
for i=1:airport_count
    for k=1:numArrays
        sumRR2 = sumRR2 + p_n(i)*S(1,i,k);
    end
end

% Define the Flight Rerouting optimization objective
RR_prob.Objective = sumRR1 + sumRR2;

% Start the loop for each affected airport
unique_vulnerable_routes_dep = unique(vulnerable_routes_dep);
for k=1:length(unique_vulnerable_routes_dep)
    % Define the flight supply for selected attacked departure airport
    d_ni = zeros(1,airport_count);
    selected_airport = unique_vulnerable_routes_dep(k);
    selected_index = vulnerable_routes_dep == unique_vulnerable_routes_dep(k);
    selected_index_complete = horzcat(att_col(selected_index),att_row(selected_index));
    lost_flights_list = zeros(1,length(selected_index_complete(:,1)));
    for i=1:length(selected_index_complete(:,1))
        lost_flights_list(i) = lost_flights_list(i) + Nominal_Y(selected_index_complete(i), ...
            selected_index_complete(i+length(selected_index_complete(:,1))));
        selected_airport2(i) = airport_unique_list(selected_index_complete(i+length(selected_index_complete(:,1))));
        selected_index2 = airport_unique_list == selected_airport2(i);
        d_ni(selected_index2) = -lost_flights_list(i);
    end
    selected_index3 = airport_unique_list == unique_vulnerable_routes_dep(k);
    d_ni(selected_index3) = sum(lost_flights_list);
    d_n(k,:) = d_ni;
end

% Set up the Flight Rerouting optimization constraints
flow_constr1 = optimconstr(numArrays,airport_count);
for t=1:numArrays
    for i=1:airport_count
        flow_constr1(t,i) = sum(Y(i,:,t)) - sum(Y(:,i,t)) - S(1,i,t) <= d_n(t,i);
    end
end

% Define lower and upper limits on traffic
transpose_Y = Y.*0;
for k=1:numArrays
    transpose_Y(:,:,k) = transpose(Y(:,:,k));
end
upperlimit_constr1 = optimconstr(airport_count,airport_count,1);
upperlimit_constr1(:,:,1) = sum(Y(:,:,1:numArrays),3) + ...
                            sum(transpose_Y(:,:,1:numArrays),3) <= u_ij(:,:);

% Define the Flight Rerouting optimization constraints
RR_prob.Constraints.consRR1 = flow_constr1;
RR_prob.Constraints.consRR2 = -S <= 0;
RR_prob.Constraints.consRR3 = upperlimit_constr1;

% Performe the optimization using a mixed-integer linear programming algorithm
RR_options = optimoptions('intlinprog','Display','final');
[YSsol,YSfval,YSexitflag,YSoutput] = solve(RR_prob,RR_options);
Solved_Y = sum(YSsol.Y(:,:,:),3);
Solved_S = sum(YSsol.S(:,:,:),3);

% Transform Solved_Y into an upper triangular matrix
for i=1:airport_count
    for j=1:airport_count
        if i>j && Solved_Y(i,j)>0
            Solved_Y(j,i) = Solved_Y(j,i) + Solved_Y(i,j);
            Solved_Y(i,j) = 0;
        end
    end
end

% Print out the routes used for rerouting
for k=1:numArrays
    [rer_col, rer_row] = find(YSsol.Y(:,:,k));
    rerouting_routes_dep = strings([length(find(YSsol.Y(:,:,k))),1]);
    rerouting_routes_arr = strings([length(find(YSsol.Y(:,:,k))),1]);
    for n=1:length(find(YSsol.Y(:,:,k)))
        rerouting_routes_dep(n) = char(airport_unique_list(rer_col(n)));
        rerouting_routes_arr(n) = char(airport_unique_list(rer_row(n)));
    end
    rerouting_routes = [rerouting_routes_dep rerouting_routes_arr];
    fprintf('The routes untilized for rerouting in response to attack no. %d are:\n', k)
    disp(rerouting_routes)
    clearvars rer_col rer_row rerouting_routes_dep rerouting_routes_arr
end

% Plot rerouted traffic map
figure(6)
image(Solved_Y,'CDataMapping','scaled')
colormap(flipud(gray))
title('Rerouted Traffic')
colorbar

% Plot non-rerouted traffic map
figure(7)
image(Solved_S,'CDataMapping','scaled')
colormap(flipud(gray))
title('Non-rerouted Traffic')
colorbar

toc
