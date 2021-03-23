n = 2;
m = 2;
% A = [1 2; 3 4];
% B = [5 6; 7 8];
% Q = 2*eye(n);
% R = eye(m);
A = randi([1, 100], n, n);
B = randi([1, 100], n, m);
Q = randi([1, 10], n, n);
Q = Q'*Q;
R = randi([1, 10], m, m);
R = R'*R;
P = idare(A, B, Q, R);
K = -inv(R+B'*P*B)*B'*P*A;
G = B*inv(R)*B';
Phi = A+B*K;
T = eye(n^2) - kron(Phi', Phi');
l = inv(norm(inv(T), 2));

syms q r
q = 1e-1;
r = 1e-1;
dQ = q*Q;
dR = r*R;
dG = B*inv(R+dR)*B'-G;
Pnew = idare(A, B, Q + dQ, R + dR);

n = @(A) norm(A, 'fro');
% left = l^2/(n(G)*n(A)^2) - 4*n(Phi)^2*n(P)^2*n(G)*n(dG)/n(G) - 4*l*n(P)*n(dG)/n(G) - ...
%     4*n(dQ)*n(dG)/n(G) - 4*n(dQ)*n(Phi)^2;
% left = ((2*n(P)*n(dG)*n(A)^2)/l - 1)^2 - (4*(n(dG)*n(A)^2 + n(G)*n(Phi)^2)*(n(dG)*n(A)^2*n(P)^2 + n(dQ)))/l^2;
% vpa(left, 3)

% a0 = (n(dQ) + n(A)^2*n(P)^2*n(dG))/l;
% a1 = (2*n(A)^2*n(P)*n(dG))/l;
% a2 = (n(Phi)^2*n(G) + n(A)^2*n(dG))/l;
% left = (1-a1)^2 - 4*a0*a2;
% vpa(left, 3)
% norm(Pnew - P)

KQF = norm(inv(T), 2);
KSF = norm(inv(T)*kron(Phi'*P, Phi'*P), 2);
kq = KQF*n(Q)/n(P);
ks = KSF*n(G)/n(P);

norm(Pnew-P)
norm(dQ)*1/(max(eig(A)))^2
KQF*(1+n(P*Phi))^2*max(n(dQ), n(dR))