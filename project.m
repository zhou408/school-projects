disp('Solving a NLP with Equality Constraints Using Local SQP Algorithm and BFGS Quasi Newton Method ');
%record program start time
tic;
%the objective function
obj = @(x1,x2) ((1-x1)^2);
%define the starting point
x1 = 2;
x2 = 1;
%define intial value of lambda(the lagrange multiplier for the equality constraint)
lambda = 1;
%define intial value of Bk (approximation of hessian matrix using BFGS) 
Bk =  [1 0;0 1];
% intial setting for the while until loop
go = true;
% define the number of iterations
iterations = 0;
while go
    %update iteration times
    iterations = iterations +1;
    fprintf('itration = %d',iterations);
    %update function value and gradient 
    fk = obj(x1,x2)
    gradfk = [-2*(1-x1); 0];
    %update constraint value and jacobian
    ck = 10*(x2-x1^2);
    Ak = [-20*x1 10];
    %define and solve the linear system
    A = [Bk -transpose(Ak);Ak 0];
    B = [-gradfk;-ck];
    C = linsolve(A,B);
    dk = C(1:2,:);
    %update lambda k and Xk 
    lambda = C(3);
    x1old = x1;
    x2old = x2;
    x1=x1+dk(1,:)
    x2=x2+dk(2,:)
    %update Bk(the Hessian approximater)
    sk = [x1-x1old;x2-x2old];
    yk = [2*x1-20*lambda^x1; -10*lambda]-[2*x1old-20*lambda^x1old; -10*lambda];
    Bk = Bk-(Bk*sk*transpose(sk)*Bk)/(transpose(sk)*Bk*sk)+(yk*transpose(yk))/(transpose(yk)*sk);
    %check if the KKT is satisfied: if satisfied, end while loop
    if ([20*lambda+2 0;0 0]*dk+[2*x1-2;0]-transpose(Ak)*lambda==[0;0]) & (Ak*dk+ck==0)
        %terminate the program, make conclusions
        go = false;
        disp('KKT is met, the program terminates successfully');
        fprintf('total number of iterations =%d',iterations);
        fprintf('\n');
        disp('the optimal solution x* is');
        disp([x1 x2]);
        fprintf('the optimal objective function value f(x*) is %d',fk);
    end
end
fprintf('\n');
%record program end time
toc;