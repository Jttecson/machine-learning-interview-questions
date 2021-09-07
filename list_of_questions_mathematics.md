# Mathematics - List of questions

## Linear Algebra
1. What is broadcasting in connection to Linear Algebra?
1. What are scalars, vectors, matrices, and tensors?
- Scalars are constants. Vectors are columns of n scalars. Matrices are m x n dimensions, a 2D grid of numbers. Tensors generalize this to be made up of any number of dimensions, from a 3D matrix, to vectors are 1D tensors.
3. What is Hadamard product of two matrices?
- Element-wise multiplication
5. What is an inverse matrix?
- Inverse of a matrix is a unique matrix such that the matrix multiplied by its inverse gives the identity matrix. Only for square matrices.
7. If inverse of a matrix exists, how to calculate it?
- Check determinant nonzero.
- Can row reduce into identity matrix and apply same reductions to the identity matrix. 
- Can break into adjugate/adjoint matrix. Multiple each element by its minor matrix determinant. Then apply cofactor checkerboard of signs. Then transpose. This is adjoint, then divide by determinant.
9. What is the determinant of a square matrix? How is it calculated (Laplace expansion)? What is the connection of determinant to eigenvalues?
- If triangular, multiply the diagonal elements. In general, can do Laplace expansion, or cofactor expansion - take each element multiplied by cofactor.
11. Discuss span and linear dependence.
- span of a set of vectors is the space of all vectors which can be expressed as a linear combination of the vectors in the set. Linear dependence means not linear independent - a linear dependent set contains a vector which can be expressed as a linear combination of all the other vectors in the set.
13. What is Ax = b? When does Ax =b has a unique solution? 
- Ax = b is a matrix equation. Matrix times vector = vector. Usually in this notation we have A, b and we want to find x. It has a unique solution when A iff every column is a pivot column - columns are lin. independent. rank(A)=n.
15. In Ax = b, what happens when A is fat or tall?
- If A is not square, then it has either zero or infinite solutions. If tall, dim(span(column space)) is likely less than m, the dimensions of vector b. So likely no solution. If fat, then column space more likely spans R^m, thus more likely to be infinite solutions.
17. When does inverse of A exist?
- A is square and nonzero determinant. Equivalently linearly independent columns or rows. rank(A) = n
19. What is a norm? What is L1, L2 and L infinity norm?
- A function from vector space to the nonnegative real numbers. It commutes with scaling, is only zero at origin, and follows triangle inequality. P(x+y) <= p(x) + p(y). L1 or Manhattan norm is sum over all dimensions. L2 or Euclidean is sqrt of sum of squares. L infinity is the max element (absolute value).
21. What are the conditions a norm has to satisfy?
- above.
23. Why is squared of L2 norm preferred in ML than just L2 norm?
- Don't have to calculate square root, which is computationally expensive. Derivative is easier to calculate for stuff like gradient descent. Can calculate sum of squares by just x^Tx which has many optimizations.
25. When L1 norm is preferred over L2 norm?
- Lasso will shrink coefficients to zero, removing that feature. Good for feature selection if we have a large feature space. L2 will not since the cost term becomes so small.
27. Can the number of nonzero elements in a vector be defined as L0 norm? If no, why?
- Doesn't associate with scalar multiplication.
29. What is Frobenius norm?
30. What is a diagonal matrix? (D_i,j = 0 for i != j)
31. Why is multiplication by diagonal matrix computationally cheap? How is the multiplication different for square vs. non-square diagonal matrix?
32. At what conditions does the inverse of a diagonal matrix exist? (square and all diagonal elements non-zero)
33. What is a symmetrix matrix? (same as its transpose)
34. What is a unit vector?
35. When are two vectors x and y orthogonal? (x.T * y = 0)
36. At R^n what is the maximum possible number of orthogonal vectors with non-zero norm?
- n
38. When are two vectors x and y orthonormal? (x.T * y = 0 and both have unit norm)
39. What is an orthogonal matrix? Why is computationally preferred? (a square matrix whose rows are mutually orthonormal and columns are mutually orthonormal.)
40. What is eigendecomposition, eigenvectors and eigenvalues?
- AKA Spectral Decomposition. Breaking down matrix into A, U (eigenvector matrix) and Y (eigenvalue diag matrix). Then AU=YU, A=U^{-1}YU. Matrix exponentiation is vastly simplified. Goes from log2(p) matrix multiplications to constant 2 matrix multiplications, for A^p.
42. How to find eigen values of a matrix?
- Solve Ax=yx => (A-yI)x = 0.
44. Write the eigendecomposition formula for a matrix. If the matrix is real symmetric, how will this change?
- For a real symmetric matrix, the eigenvectors can be found to be orthogonal.
45. Is the eigendecomposition guaranteed to be unique? If not, then how do we represent it?
- No, we eigenvalues can be the same. By convention, order eigenvalues in Y in descending order.
47. What are positive definite, negative definite, positive semi definite and negative semi definite matrices?
- Positive definite - all positive eigenvalues. Negative - all negative. semi - can be zero.
- If A is positive definite, then x^TAx >= 0 for all x.
49. What is SVD? Why do we use it? Why not just use ED?
- SVD is a form of data reduction, taylored to the specific problem/data of interest. Takes some data matrix X, represents X=UEV^T. U, V are unitary square, sigma E is diagonal rectangular nxm (singular values), and ordered in decreasing order. U contains info about the column space of X, V the row space. Ordered by significance using singular values sigma_1, .. etc.
51. Given a matrix A, how will you calculate its SVD?
52. What are singular values, left singulars and right singulars?
53. What is the connection of SVD of A with functions of A?
54. Why are singular values always non-negative?
55. What is the Moore Penrose pseudo inverse and how to calculate it?
56. If we do Moore Penrose pseudo inverse on Ax = b, what solution is provided is A is fat? Moreover, what solution is provided if A is tall?
57. Which matrices can be decomposed by ED? (Any NxN square matrix with N linearly independent eigenvectors)
58. Which matrices can be decomposed by SVD? (Any matrix; V is either conjugate transpose or normal transpose depending on whether A is complex or real)
59. What is the trace of a matrix?
60. How to write Frobenius norm of a matrix A in terms of trace?
61. Why is trace of a multiplication of matrices invariant to cyclic permutations?
62. What is the trace of a scalar?
63. Write the frobenius norm of a matrix in terms of trace?

## Numerical Optimization
1. What is underflow and overflow? 
1. How to tackle the problem of underflow or overflow for softmax function or log softmax function? 
1. What is poor conditioning? 
1. What is the condition number? 
1. What are grad, div and curl?
1. What are critical or stationary points in multi-dimensions?
1. Why should you do gradient descent when you want to minimize a function?
1. What is line search?
1. What is hill climbing?
1. What is a Jacobian matrix?
1. What is curvature?
1. What is a Hessian matrix?

## Basics of Probability and Informaion Theory
1. Compare "Frequentist probability" vs. "Bayesian probability"?
1. What is a random variable?
1. What is a probability distribution?
1. What is a probability mass function?
1. What is a probability density function?
1. What is a joint probability distribution?
1. What are the conditions for a function to be a probability mass function?
1. What are the conditions for a function to be a probability density function?
1. What is a marginal probability? Given the joint probability function, how will you calculate it?
1. What is conditional probability? Given the joint probability function, how will you calculate it?
1. State the Chain rule of conditional probabilities.
1. What are the conditions for independence and conditional independence of two random variables?
1. What are expectation, variance and covariance?
1. Compare covariance and independence.
1. What is the covariance for a vector of random variables?
1. What is a Bernoulli distribution? Calculate the expectation and variance of a random variable that follows Bernoulli distribution?
1. What is a multinoulli distribution?
1. What is a normal distribution?
1. Why is the normal distribution a default choice for a prior over a set of real numbers?
1. What is the central limit theorem?
1. What are exponential and Laplace distribution?
1. What are Dirac distribution and Empirical distribution?
1. What is mixture of distributions?
1. Name two common examples of mixture of distributions? (Empirical and Gaussian Mixture)
1. Is Gaussian mixture model a universal approximator of densities?
1. Write the formulae for logistic and softplus function.
1. Write the formulae for Bayes rule.
1. What do you mean by measure zero and almost everywhere?
1. If two random variables are related in a deterministic way, how are the PDFs related?
1. Define self-information. What are its units?
1. What are Shannon entropy and differential entropy?
1. What is Kullback-Leibler (KL) divergence?
1. Can KL divergence be used as a distance measure?
1. Define cross-entropy.
1. What are structured probabilistic models or graphical models?
1. In the context of structured probabilistic models, what are directed and undirected models? How are they represented?
What are cliques in undirected structured probabilistic models?

## Confidence interval
1. What is population mean and sample mean?
1. What is population standard deviation and sample standard deviation?
1. Why population s.d. has N degrees of freedom while sample s.d. has N-1 degrees of freedom? In other words, why 1/N inside root for pop. s.d. and 1/(N-1) inside root for sample s.d.? (Here)
1. What is the formula for calculating the s.d. of the sample mean?
1. What is confidence interval?
1. What is standard error?

