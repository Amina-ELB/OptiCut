import ufl

def strain(v):
        r"""Calculus of the strain tensor, for elasticity law as:

        .. math::
            
            \varepsilon(u) = \frac{1}{2}(\nabla\cdot u + \nabla^{T} \cdot u).
            
            
        :param fem.Function v: The displacement field function.

        :returns: Expression of the strain tensor.
        :rtype: fem.Expression

        """
        return  (0.5 * (ufl.nabla_grad(v) + ufl.nabla_grad(v).T))
        
def stress(u,lame_mu,lame_lambda,dim):
        r"""Calculus of the stress tensor, for elasticity law as

        .. math::
            
            \sigma(u) = \lambda  \nabla\cdot u * \text{Id} + 2\mu * \varepsilon(u) 
            
            

        with :math:`\varepsilon(u)` computed with the function :func:`mecanics_tool.strain`.

        :param fem.Function v: The displacement field function.
        :param float lame_mu: the :math:`\mu` Lame coefficient.
        :param float lame_lambda: the :math:`\lambda` Lame coefficient.
        :param int dim: the dimension of the displacement field.
        
        :returns: Expression of the stress tensor.
        :rtype: fem.Expression
        
        """
        return lame_lambda * ufl.nabla_div(u) * ufl.Identity(dim) + 2*lame_mu*strain(u)

def lame_compute(E,v):
    r"""Calculus of the lame coefficient with Young modulus and Poisson coefficient as :

    .. math::
        \begin{align}
        \mu &= \frac{E}{2(1+\nu)}  \\
        \lambda &= \frac{E\nu}{(1+\nu)(1-2\nu)}
        \end{align}

    with :math:`\sigma(u)` computed with the function :func:`mecanics_tool.stress`.
        

    :param float E: The Young modulus.
    :param float v: The Poisson coefficient.
    
    :returns: Lame :math:`\mu` and Lame :math:`\lambda` coefficients.
    :rtype: float, float
        
    """
    lame_mu = E / (2.0 * (1.0 + v))
    lame_lambda = E * v / ((1.0 + v) * (1.0 - 2.0 * v))    
    return lame_mu, lame_lambda

def von_mises(u,lame_mu,lame_lambda,dim):
    r"""Calculus of the Von Mises stress value:

    .. math::

        \sigma_{VM} = \sigma(u) - \frac{1}{3}\text{Tr}(\sigma(u))\text{Id}


    with :math:`\sigma(u)` compute with the function :func:`mecanics_tool.stress`.
    

    :param fem.Function u: The displacement field function.
    :param float lame_mu: The Lame :math:`\mu` coefficient.
    :param float lame_lambda: The lame :math:`\lambda` coefficient.
    :param float dim: The dimension of the displacement field.
    
    :returns: The value of the Von Mises stress constraint.
    :rtype: fem.Function
    
    """
    s = stress(u,lame_mu,lame_lambda,dim) -(1./3)*ufl.tr(stress(u,lame_mu,lame_lambda,dim))*ufl.Identity(dim)
    r =(2./3)*ufl.inner(s,s)
    return ufl.sqrt(r)


