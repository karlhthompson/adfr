%% Import data from spreadsheet

tic
if exist('APMR','var')
else
    % Import the data
    [~, ~, raw] = xlsread('C:\Users\karlh\Box\Documents\MATLAB\APM-Report-10892.xls','APM-Report-10892','A8:Q4659');
    stringVectors = string(raw(:,[1,2,17]));
    stringVectors(ismissing(stringVectors)) = '';
    raw = raw(:,[3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
    
    % Create output variable
    data = reshape([raw{:}],size(raw));
    
    % Create table
    APMR = table;
    
    % Allocate imported array to column variable names
    APMR.Departure = stringVectors(:,1);
    APMR.Arrival = stringVectors(:,2);
    APMR.Flight = data(:,1);
    
%     % Transform table into array
%     APMR = table2array(APMR);
%     total_num_flights = str2double(APMR(end,3));
%     APMR(29:end,:) = []; % Choose how much data to include
%     
%     % Clear temporary variables
%     clearvars data raw stringVectors;
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
all_departure_flights = categorical(APMR(:,1));
all_arrival_flights = categorical(APMR(:,2));
all_flight_numbers = str2double(APMR(:,3));

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

% Figure out the hub airports
hub_count = airport_count;
n1 = 1;
while hub_count>=0.125*airport_count
    hub_indexes = dep_air_num >= n1;
    hub_names = dep_airport_unique_list(hub_indexes);
    hub_count = length(hub_names);
    if hub_count==0
        n1 = n1-1;
        hub_indexes = dep_air_num >= n1;
        hub_names = dep_airport_unique_list(hub_indexes);
        hub_count = length(hub_names);
        break
    else
        n1 = n1+1;
    end
end
clearvars n1;

% % Remove all non-hub departure airports
% total_hub_indexes = ismember(categorical(APMR(:,1)),hub_names);
% hub_departure_flights = all_departure_flights(total_hub_indexes);
% hub_arrival_flights = all_arrival_flights(total_hub_indexes);
% hub_flight_numbers = all_flight_numbers(total_hub_indexes);
% total_num_flights_hubs = sum(hub_flight_numbers);

% Plot the initial network state
C = convertStringsToChars(APMR(:,1));
D = convertStringsToChars(APMR(:,2));
F = str2double(APMR(:,3));
G = convertStringsToChars(unique(vertcat(APMR(:,1), APMR(:,2))));
E = digraph(C,D,F,G);
figure(1)
plot(E,'Layout','force','EdgeLabel',E.Edges.Weight)
title('Initial Network State')
clearvars C D E F G;

%% Set up the COMBINED ATTACKER-DEFENDERz optimization problem

% Create an index matrix for all airports
index_matrix = 1:1:airport_count;

% Replace the airport name values in APMR by their indexes
APMR_num = APMR;
for n2=1:airport_count
    a1 = airport_unique_list(n2);
    a2 = find(categorical(APMR)==a1);
    APMR_num(a2)=n2;
end
APMR_num = str2double(APMR_num);
clearvars a1 a2 n2;

% Define the network state
x_ij = ones(airport_count,airport_count,1);
for n3=1:length(APMR_num)
    x_ij(APMR_num(n3,1),APMR_num(n3,2)) = 0;
end
clearvars n3;

% Define the per-unit cost of traversing an arc
c_ij = ones(airport_count,airport_count,1);

% Define the upper bound on total undirected flow on an edge
u_ij = ones(airport_count,airport_count,1) .* 300;

% Define the per-unit penalty cost of traversing an arc if damaged
q_ij = ones(airport_count,airport_count,1) .* 400; %200 can be any large enough number

% Define the per-unit penalty cost for demand shortfall
p_ij = ones(airport_count,airport_count,1) .* 999;

% Nominal flow on each route
Nominal_Y = zeros(airport_count,airport_count);
for n=1:length(APMR_num)
    if APMR_num(n,1) <= APMR_num(n,2)
        Nominal_Y(APMR_num(n,1),APMR_num(n,2)) = ...
            Nominal_Y(APMR_num(n,1),APMR_num(n,2)) + APMR_num(n,3);
    else
        Nominal_Y(APMR_num(n,2),APMR_num(n,1)) = ...
            Nominal_Y(APMR_num(n,2),APMR_num(n,1)) + APMR_num(n,3);
    end
end

% Define the cost to 'break' each route
r_ij = ones(airport_count,airport_count,1);
for n=1:length(APMR_num)
    if ismember(APMR(n,1),hub_names)
        r_ij(APMR_num(n,1),:) = 2;
        r_ij(:,APMR_num(n,1)) = 2;
    end
end
r_ij(Nominal_Y==0) = 99999; % Ensuring the attacker doesn't attack non-existent routes

% Define the available attack budget
attack_budget = 2;

% Define the cost to protect existing routes
h_ij = ones(airport_count,airport_count,1);
for n=1:length(APMR_num)
    if ismember(APMR(n,1),hub_names)
        h_ij(APMR_num(n,1),:) = 2;
        h_ij(:,APMR_num(n,1)) = 2;
    end
end
h_ij(Nominal_Y==0) = 99999; % Ensuring the defender doesn't defend non-existent routes

% Define the available defence budget
defence_budget = 2;

% Plot flight distribution map
figure(2)
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

sumAD1 = 0;
for i=1:airport_count
    for j=1:airport_count
        sumAD1 = sumAD1 + sum(((c_ij(i,j)+q_ij(i,j)*(1-X(i,j)-W(i,j)+...
                 XW(i,j)))*Nominal_Y(i,j)) + ((c_ij(j,i)+ ...
                 q_ij(j,i)*(1-X(i,j)-W(i,j)+XW(i,j)))*Nominal_Y(i,j)));
    end
end

% Define the AD Model optimization objective
AD_prob.Objective = sumAD1;

% Prepare the optimization constraints
% Attack budget constraint
att_budget_constr = sum(sum(r_ij.*X)) == attack_budget;
% Defence budget constraint
def_budget_constr = sum(sum(h_ij.*W)) == defence_budget;
% Product constraints
for i=1:airport_count
    for j=1:airport_count
        product_constr1(i,j) = XW(i,j) <= X(i,j);
        product_constr2(i,j) = XW(i,j) <= W(i,j);
        product_constr3(i,j) = W(i,j) - XW(i,j) <= 1 - X(i,j);
    end
end

% Define the optimization constraints
AD_prob.Constraints.consAD1 = att_budget_constr;
AD_prob.Constraints.consAD2 = def_budget_constr;
AD_prob.Constraints.consAD3 = product_constr1;
AD_prob.Constraints.consAD4 = product_constr2;
AD_prob.Constraints.consAD5 = product_constr3;

% Performe the optimization using a mixed-integer linear programming algorithm
AD_options = optimoptions('intlinprog','Display','final');
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
figure(3)
image(Solved_X,'CDataMapping','scaled')
colormap(flipud(gray))
title('Attacked Routes')

% Plot the updated network state
APMR2 = APMR;
for n=1:length(att_col)
    for k=1:length(APMR_num)
        if APMR_num(k,1:2)==[att_col(n) att_row(n)]
            APMR2(k,:) = 0;
        end
    end
end
A = APMR2(:,1); A(A=="0")=[];
B = APMR2(:,2); B(B=="0")=[];
C = convertStringsToChars(A);
D = convertStringsToChars(B);
E = digraph(C,D);
figure(4)
plot(E,'Layout','force')
title('Updated Network State')
clearvars A B C D E APMR2;

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

%% Setting up the Flight Rerouting optimization problem

% Update X solution variables
if exist('Solved_WB','var')
    X_updated = X_updated - Solved_WB;
else
    X_updated = Solved_X + x_ij;
end

% Define solution variables
% Rerouted flow through existing network routes
numArrays = length(unique(vulnerable_routes_dep));
Y = optimvar('Y',airport_count,airport_count,numArrays,'Type','integer','LowerBound',0);
% Flight shortfall at each airport
S = optimvar('S',1,airport_count,numArrays,'Type','integer','LowerBound',0);

% Define the optimization problem
RR_prob = optimproblem('ObjectiveSense','minimize');

sumRR1 = 0;
for i=1:airport_count
    for j=1:airport_count
        for k=1:numArrays
            sumRR1 = sumRR1 + sum(((c_ij(i,j)+q_ij(i,j)*X_updated(i,j))*Y(i,j,k)) + ...
                ((c_ij(j,i)+q_ij(j,i)*X_updated(i,j))*Y(j,i,k)));
        end
    end
end

sumRR2 = 0;
p_n = ones(1,airport_count,1) .* 9999;
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
for k=1:numArrays
    sum10 = 0;
    for n=1:airport_count % this has to be for outgoing traffic from nodes
        sum10 = 0;
        for j=1:airport_count
            if X_updated(n,j,:) == 0
                sum10 = sum10 + Y(n,j,k) - Y(j,n,k);
                sum10_constr(k,n) = sum10;
            end
            if X_updated(n,:,:) == 1
                sum10_constr(k,n) = 0*Y(n,j,k);
            end
        end
    end
end

for k=1:numArrays
    sum11 = 0;
    for n=1:airport_count % this has to be for ingoing traffic into the node
        sum11 = 0;
        for i=1:airport_count
            if X_updated(i,n,:) == 0
                sum11 = sum11 + Y(i,n,k) - Y(n,i,k);
                sum11_constr(k,n) = sum11;
            end
            if X_updated(:,n,:) == 1
                sum11_constr(k,n) = 0*Y(n,j,k);
            end
        end
    end
end

flow_constr1 = optimconstr(numArrays,airport_count);
for i=1:airport_count
    for t=1:numArrays
        flow_constr1(t,i) = sum10_constr(t,i) - sum11_constr(t,i) - S(1,i,t) <= d_n(t,i);
    end
end

symmetry_constr1 = optimconstr(airport_count,airport_count,numArrays);
for k=1:numArrays
    symmetry_constr1(:,:,k) = -Y(:,:,k) -transpose(Y(:,:,k)) <= 0;
end

% Define the Flight Rerouting optimization constraints
RR_prob.Constraints.consRR1 = flow_constr1;
RR_prob.Constraints.consRR2 = symmetry_constr1;
RR_prob.Constraints.consRR3 = -S <= 0;

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
