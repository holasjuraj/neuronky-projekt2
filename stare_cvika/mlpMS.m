
function [h,y] = forward_pass(x, w)
	x = x(1:end-1);
	h{1} = x;
	h_net{1} = x; 
	for i = 2 : size(w,2) + 1
    	y = w{i-1} * [h{i-1};-1];
    	h{i} = logsig(y);
	end
endfunction

%nacitanie dat
clear
load train_set
load test_set

n_input = 1;
n_out = 1;

n_train = size(train_set,2);
n_test = size(test_set,2);

%vygeneruj vahy, nastav parametre
alpha = 0.1;   %  <--- uprav
layers = [ n_input, 10 , n_out ] ;     %  <--- uprav
m = [0.65, 0.55, 0.50 ]
%w(i) = rand(vstupy, vystupy + 1(bias) 
w = [];
dw = [];

for i = 2 : size(layers,2)
	w{i - 1} = rand( layers(i) ,layers(i-1) + 1);
	dw{i - 1} = zeros(size(w{i-1}));
end



errors = [];
E = 1;
ep = 0;

time = cputime;

minE = 100000;
minW = w;

while ( ep < 1000 )
  	E = 0;
   	ep++;
   	train_set = train_set(:,randperm(n_train));
	for t=1:n_train

		x = train_set(:,t);
		[h, y] = forward_pass( x, w );

		%%%%% Vyratanie targetu a chyby %%%%%%
		target = train_set(n_input + 1,t);
		e = (1/2)*(target - y )^2;

		%%%%%%%%%%%%%%%%%%%%%%%%%%
		for i = size(w,2): -1 : 1
			if i == size(w,2)
				delta{i} = target - y;
			else
				w_unbias = w{i+1}(:,1:end-1);
				%h_unbias = h{i+1}(1:end-1,:);
				delta{i} = (w_unbias'*delta{i+1}).*(h{i+1}.*(1 - h{i+1})); 	

			end
			tmp = alpha*delta{i}*([h{i};-1]');%; + m(i)*dw{i};	
			w{i} = w{i} + tmp;
			%dw{i} = tmp;
      		
		end
		%%%%%%%%%%%%%%%%%%%%%%%%%%
	   	E += e;

   	end

   	if mod(ep, 10) == 0
	   	ep
	   	E
	   	time = cputime - time
	   	time = cputime;
   	end

   	if E < minE
   		minE = E;
   		minW = w;
   	end
end

w = minW;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% testovanie
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
load full_set;

E = 0;
for j=1:n_test    
    %prechod
       x = test_set(:, j);
       % DOPLN - forward pass
       [h, y] = forward_pass(x, w);
       
   	   % vyratanie targetu
   	   target = test_set(n_input,j);
       
    % spocitava pocet spravne urcenych vzoriek
    E = E + (target-y)^2;
end
   
E = sqrt(E / n_test);

min_data = min(data(2,:)');
max_data = max(data(2,:)');
E_test = E / (max_data - min_data);

E=0;
%data = (sortrows(data'))';
for j=1:size(data,2)
        
       %prechod
       x = data(:, j);
       % DOPLN - forward pass
       [h, y] = forward_pass(x, w);
       
       plot_data(j,1) = y;
       plot_data(j,2) = data(2,j);
       
       E = E + (data(2,j)-y)^2;
end
E = sqrt(E / n_test);
E_overal = E / (max_data - min_data)   

figure;
plot(plot_data)