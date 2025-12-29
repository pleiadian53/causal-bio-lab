"""
Base classes for Structural Causal Models (SCMs).

A Structural Causal Model consists of:
1. Endogenous variables (V) - variables in the system
2. Exogenous variables (U) - noise/unobserved factors
3. Structural equations (F) - how V are generated from U and other V
"""

from typing import Dict, Callable, Optional, Any, List, Tuple
import numpy as np
from scipy import stats
from dataclasses import dataclass, field


@dataclass
class SCMVariable:
    """Represents a variable in an SCM."""
    name: str
    equation: Callable
    parents: List[str] = field(default_factory=list)
    noise_dist: Optional[Any] = None
    
    def __post_init__(self):
        """Set default noise distribution if not provided."""
        if self.noise_dist is None:
            self.noise_dist = stats.norm(0, 1)


class StructuralCausalModel:
    """
    Base class for Structural Causal Models.
    
    An SCM defines how variables are generated from exogenous noise
    and other variables through structural equations.
    
    Parameters
    ----------
    variables : Dict[str, SCMVariable]
        Dictionary mapping variable names to SCMVariable objects
    topological_order : Optional[List[str]]
        Order in which to evaluate variables (respects causal ordering)
        If None, will be inferred from parent relationships
    
    Examples
    --------
    >>> # Simple linear SCM: X -> Y
    >>> scm = StructuralCausalModel({
    ...     'X': SCMVariable('X', lambda u_x: u_x),
    ...     'Y': SCMVariable('Y', lambda x, u_y: 2*x + u_y, parents=['X'])
    ... })
    >>> data = scm.sample(n_samples=1000)
    """
    
    def __init__(
        self,
        variables: Dict[str, SCMVariable],
        topological_order: Optional[List[str]] = None
    ):
        self.variables = variables
        self.topological_order = topological_order or self._infer_topological_order()
        self._validate()
    
    def _infer_topological_order(self) -> List[str]:
        """Infer topological ordering from parent relationships."""
        # Simple topological sort using Kahn's algorithm
        in_degree = {name: len(var.parents) for name, var in self.variables.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            
            # Find children of this node
            for name, var in self.variables.items():
                if node in var.parents:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
        
        if len(order) != len(self.variables):
            raise ValueError("Graph contains a cycle - not a valid DAG")
        
        return order
    
    def _validate(self):
        """Validate the SCM structure."""
        # Check that all parent variables exist
        for name, var in self.variables.items():
            for parent in var.parents:
                if parent not in self.variables:
                    raise ValueError(f"Parent '{parent}' of '{name}' not found in variables")
    
    def sample(
        self,
        n_samples: int = 1000,
        random_seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate observational samples from the SCM.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        random_seed : Optional[int]
            Random seed for reproducibility
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping variable names to arrays of samples
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        samples = {}
        
        # Generate samples in topological order
        for var_name in self.topological_order:
            var = self.variables[var_name]
            
            # Sample exogenous noise
            u = var.noise_dist.rvs(n_samples)
            
            # Get parent values
            parent_values = [samples[parent] for parent in var.parents]
            
            # Evaluate structural equation
            if len(parent_values) == 0:
                # Root node - only depends on noise
                samples[var_name] = var.equation(u)
            else:
                # Has parents
                samples[var_name] = var.equation(*parent_values, u)
        
        return samples
    
    def intervene(
        self,
        interventions: Dict[str, Any]
    ) -> 'StructuralCausalModel':
        """
        Apply interventions (do-operator) to create a new mutilated SCM.
        
        This implements Pearl's do-operator by:
        1. Removing incoming edges to intervened variables
        2. Setting those variables to fixed values
        
        Parameters
        ----------
        interventions : Dict[str, Any]
            Dictionary mapping variable names to intervention values
        
        Returns
        -------
        StructuralCausalModel
            New SCM with interventions applied
        
        Examples
        --------
        >>> # Intervene on X, setting it to 1.5
        >>> scm_do_x = scm.intervene({'X': 1.5})
        >>> data_do_x = scm_do_x.sample(1000)
        """
        new_variables = {}
        
        for name, var in self.variables.items():
            if name in interventions:
                # Create new variable with constant equation
                intervention_value = interventions[name]
                new_variables[name] = SCMVariable(
                    name=name,
                    equation=lambda u, val=intervention_value: np.full_like(u, val),
                    parents=[],  # Remove all parents
                    noise_dist=var.noise_dist
                )
            else:
                # Keep original variable
                new_variables[name] = var
        
        return StructuralCausalModel(new_variables, self.topological_order)
    
    def counterfactual(
        self,
        observed: Dict[str, float],
        intervention: Dict[str, Any],
        query: str
    ) -> float:
        """
        Compute counterfactual query using abduction-action-prediction.
        
        This implements Pearl's three-step counterfactual computation:
        1. Abduction: Infer exogenous variables U from observed data
        2. Action: Apply intervention to create mutilated SCM
        3. Prediction: Compute query variable under intervention with inferred U
        
        Parameters
        ----------
        observed : Dict[str, float]
            Observed values of variables
        intervention : Dict[str, Any]
            Intervention to apply (counterfactual world)
        query : str
            Variable to query in counterfactual world
        
        Returns
        -------
        float
            Counterfactual value of query variable
        
        Examples
        --------
        >>> # "What would Y be if we had set X=2, given we observed X=1, Y=3?"
        >>> y_cf = scm.counterfactual(
        ...     observed={'X': 1, 'Y': 3},
        ...     intervention={'X': 2},
        ...     query='Y'
        ... )
        
        Notes
        -----
        This is a simplified implementation that assumes:
        - Exogenous variables can be uniquely determined (identifiable)
        - Structural equations are invertible for abduction
        """
        # Step 1: Abduction - infer exogenous variables
        exogenous = self._abduct(observed)
        
        # Step 2: Action - create intervened SCM
        scm_intervened = self.intervene(intervention)
        
        # Step 3: Prediction - compute query under intervention with inferred U
        result = scm_intervened._predict_with_exogenous(exogenous, query)
        
        return result
    
    def _abduct(self, observed: Dict[str, float]) -> Dict[str, float]:
        """
        Infer exogenous variables from observed endogenous variables.
        
        This is the "abduction" step in counterfactual reasoning.
        
        Note: This is a simplified implementation. In general, abduction
        requires solving the structural equations for U, which may not
        always have a unique solution.
        """
        exogenous = {}
        
        # Work through variables in topological order
        for var_name in self.topological_order:
            if var_name not in observed:
                continue
            
            var = self.variables[var_name]
            obs_value = observed[var_name]
            
            # Get parent values (either observed or computed)
            parent_values = []
            for parent in var.parents:
                if parent in observed:
                    parent_values.append(observed[parent])
                else:
                    raise ValueError(f"Cannot abduct: parent '{parent}' not observed")
            
            # Infer exogenous variable by inverting the structural equation
            # This is problem-specific and may require custom implementation
            # For now, we store the observed value and will handle in subclasses
            exogenous[var_name] = obs_value
        
        return exogenous
    
    def _predict_with_exogenous(
        self,
        exogenous: Dict[str, float],
        query: str
    ) -> float:
        """
        Predict query variable using specific exogenous values.
        
        This is the "prediction" step in counterfactual reasoning.
        """
        values = {}
        
        # Evaluate in topological order
        for var_name in self.topological_order:
            var = self.variables[var_name]
            
            # Get parent values
            parent_values = [values[parent] for parent in var.parents]
            
            # Use stored exogenous value or default
            if var_name in exogenous:
                u = exogenous[var_name]
            else:
                u = 0.0  # Default
            
            # Evaluate equation
            if len(parent_values) == 0:
                values[var_name] = var.equation(u)
            else:
                values[var_name] = var.equation(*parent_values, u)
            
            if var_name == query:
                return values[var_name]
        
        raise ValueError(f"Query variable '{query}' not found")
    
    def get_dag(self) -> Dict[str, List[str]]:
        """
        Get the DAG structure as an adjacency list.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping each variable to its children
        """
        dag = {name: [] for name in self.variables}
        
        for name, var in self.variables.items():
            for parent in var.parents:
                dag[parent].append(name)
        
        return dag
    
    def __repr__(self) -> str:
        """String representation of the SCM."""
        lines = ["StructuralCausalModel("]
        lines.append(f"  Variables: {list(self.variables.keys())}")
        lines.append(f"  Topological order: {self.topological_order}")
        lines.append("  Structural equations:")
        for name in self.topological_order:
            var = self.variables[name]
            if var.parents:
                lines.append(f"    {name} := f({', '.join(var.parents)}, U_{name})")
            else:
                lines.append(f"    {name} := f(U_{name})")
        lines.append(")")
        return "\n".join(lines)
