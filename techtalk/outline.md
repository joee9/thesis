# Slides

1. Introduction
2. What are neutron stars? (maybe get a bit more technical)
3. What are equations of state?
4. How can we use equations of state to make predictions?
    - TOV equations (solving initial value problem, initial conditions, outer boundary conditions, RK4)
    - Mass radius diagrams and examples (include units on axes and shrink plots)
    - implementation in Python: use numpy, scipy integration, rootfinding, terminate once edge of star has been met, interpolation function for tabulated values)
    - introduce SLy and the predictions that it makes: a "realistic" equation of state
5. An example and its predictions (QHD-I):
    - Brief slide about theory
    - Lagrangian and the associated fields
    - Derivation of equations of motion
    - Reduced mean field theory approximations; expectation values and the resulting equations
    - reduced Lagrangian and how it leads to EMT and equation of state (expectation values are beyond the scope of this project)
    - resulting equations; $k_f$ as a free parameter (loop variable)
    - methodology for calculating equation of state: loop over $k_f$, calculate $\phi_0$ and $V_0$, then $\epsilon$ and $P$.
    - results of using this equation of state within the TOV equations; the predictions that it makes
    