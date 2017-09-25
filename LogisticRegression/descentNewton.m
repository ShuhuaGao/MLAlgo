function wstar = descentNewton(func, gradient, hessian, w0, epsilon)
%DESCENTNEWTON Newton's method for descent optimization
%wstar = descentNewton(f, w0)
%INPUT:
%func, gradient and hessian are three function handles to compute the corresponding values
%w0: initial value of w for optimization
%epsilon: tolerance to stop the iteration
%wstar: the minimizer found by the algorithm

w = w0;
nIteration = 0;
while true
    nIteration = nIteration + 1;
    %1. compute the Newton step and decrement
    g = gradient(w);
    h = hessian(w);
    ns = -h\g; % Newton step
    decrement = abs(g'*ns);
    %2. stop criterion
    if decrement <= epsilon
        break;
    end
    %3. line search to choose step size t
%     alpha = 0.3;
%     beta = 0.5;
%     t = 1;
%     while func(w + t*ns) > func(w) + alpha*t*g'*ns
%         t = beta * t;
%     end
    t = 0.15;
    %4. update
    w = w + t*ns;
end
wstar = w;
fprintf('Iteration number = %d\n', nIteration);
end

