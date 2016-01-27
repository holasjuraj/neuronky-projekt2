%nacitanie dat
clear
load iris

[n_input,n_data] = size(data);
n_out = max(data(n_input, :));
data = data(:, randperm(n_data));

% rozdel data na trenovaci a na testovaci set
train_set = data(:, 1:120);
test_set = data(:, 121:150);

n_train = size(train_set,2);
n_test = size(test_set,2);

%vygeneruj vahy, nastav parametre
alpha =  0.1; % DOPLN
n_hid =  10; % DOPLN

w_hid = rand(n_hid, 5); % DOPLN
w_out = rand(n_out, n_hid + 1); % DOPLN

errors = [];
E = 1;
%while (E > 0.05)
for ep = 1 : 250
   % trenovanie
   train_set = train_set(:,randperm(n_train));
   for j = 1:n_train
       x = train_set(:, j);
       x(end) = -1;
       % DOPLN - forward pass
       h = [logsig(w_hid * x); -1];
       y = logsig(w_out * h);
       
       % vyratanie targetu
       target = zeros(n_out, 1);
       target(train_set(n_input, j)) = 1;
       
       %vyrataj chybu na vrstvach
       % DOPLN - backward pass + trenovanie
       sigma_out = (target - y) .* y .* (1 - y);
       w_out_unbias = w_out(:, 1:end-1);
       h_unbias = h(1:end-1);
       sigma_hid = w_out_unbias' * sigma_out .* h_unbias .* (1 - h_unbias);
       
       w_out = w_out + alpha*sigma_out*h';
       w_hid = w_hid + alpha*sigma_hid*x';       
   end
   
   % testovanie
   E = 0;
   for j=1:n_test
       x = test_set(:, j);
       x(end) = -1;
       % DOPLN - forward pass z trenovania
       h = [logsig(w_hid * x); -1];
       y = logsig(w_out * h);
       
       % vyratanie targetu
       target = zeros(n_out,1);
       target(test_set(n_input,j)) = 1;
       
       % spocitava pocet spravne urcenych vzoriek
       E = E + min(((y == max(y)) == target));
   end
    
   E = (n_test - E) / n_test;
   errors = [errors , E];
end
figure;
plot(errors);
