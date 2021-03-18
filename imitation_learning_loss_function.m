n = 2;
m = 2;
A = rand(n, n);
B = rand(n, m);
Q = rand(n, n);
Q = Q'*Q;
R = rand(m, m);
R = R'*R;
P = idare(A, B, Q, R);
K = -inv(R+B'*P*B)*B'*P*A;
x0 = rand(n, 1);

N = 5;
D = zeros(n, N*n);
D(1:n, 1:n) = eye(n);
for i=2:N
    D(1:n, i+1:i+n) = power(A+B*K, i-1)';
end
D

X0 = zeros(N*n, 1);
for i=1:N
    X0(i+1:i+n, 1) = x0;
end

X0

X = D'*x0
size(D)
size(x0)
size(X)
size(Q)
JQ = X'*Q*X