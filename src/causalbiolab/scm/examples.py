"""
Example SCMs for biological and general applications.

This module provides pre-built SCM examples for common scenarios.
"""

from typing import Optional
import numpy as np
from scipy import stats

from .base import StructuralCausalModel, SCMVariable
from .counterfactuals import LinearSCM


def simple_linear_scm() -> StructuralCausalModel:
    """
    Simple linear SCM: X -> Y
    
    Structural equations:
        X = U_X
        Y = 2*X + U_Y
    
    Returns
    -------
    StructuralCausalModel
        Simple two-variable linear SCM
    """
    variables = {
        'X': SCMVariable(
            name='X',
            equation=lambda u_x: u_x,
            parents=[],
            noise_dist=stats.norm(0, 1)
        ),
        'Y': SCMVariable(
            name='Y',
            equation=lambda x, u_y: 2*x + u_y,
            parents=['X'],
            noise_dist=stats.norm(0, 0.5)
        )
    }
    
    return StructuralCausalModel(variables)


def confounded_scm() -> StructuralCausalModel:
    """
    Confounded SCM: Z -> X, Z -> Y
    
    Structural equations:
        Z = U_Z
        X = Z + U_X
        Y = 2*X + Z + U_Y
    
    Z is a confounder affecting both X and Y.
    
    Returns
    -------
    StructuralCausalModel
        Three-variable confounded SCM
    """
    variables = {
        'Z': SCMVariable(
            name='Z',
            equation=lambda u_z: u_z,
            parents=[],
            noise_dist=stats.norm(0, 1)
        ),
        'X': SCMVariable(
            name='X',
            equation=lambda z, u_x: z + u_x,
            parents=['Z'],
            noise_dist=stats.norm(0, 0.5)
        ),
        'Y': SCMVariable(
            name='Y',
            equation=lambda x, z, u_y: 2*x + z + u_y,
            parents=['X', 'Z'],
            noise_dist=stats.norm(0, 0.5)
        )
    }
    
    return StructuralCausalModel(variables)


def gene_regulation_scm() -> StructuralCausalModel:
    """
    Gene regulatory network SCM: TF -> Gene -> Protein
    
    Structural equations:
        TF = U_TF (transcription factor activity)
        Gene = sigmoid(TF) + U_Gene (gene expression)
        Protein = Gene * exp(U_Protein) (protein abundance)
    
    Returns
    -------
    StructuralCausalModel
        Gene regulation SCM with nonlinear relationships
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    variables = {
        'TF': SCMVariable(
            name='TF',
            equation=lambda u_tf: u_tf,
            parents=[],
            noise_dist=stats.norm(0, 1)
        ),
        'Gene': SCMVariable(
            name='Gene',
            equation=lambda tf, u_gene: sigmoid(tf) + 0.1 * u_gene,
            parents=['TF'],
            noise_dist=stats.norm(0, 1)
        ),
        'Protein': SCMVariable(
            name='Protein',
            equation=lambda gene, u_prot: gene * np.exp(0.2 * u_prot),
            parents=['Gene'],
            noise_dist=stats.norm(0, 1)
        )
    }
    
    return StructuralCausalModel(variables)


def drug_response_scm() -> StructuralCausalModel:
    """
    Drug response SCM with genetic modifier.
    
    Structural equations:
        Genotype = U_Genotype (binary: 0 or 1)
        DrugMetabolism = Genotype * 0.5 + U_Metabolism
        Response = 2*DrugDose - DrugMetabolism + U_Response
    
    Genotype affects drug metabolism, which affects response.
    
    Returns
    -------
    StructuralCausalModel
        Drug response SCM
    """
    variables = {
        'Genotype': SCMVariable(
            name='Genotype',
            equation=lambda u_g: (u_g > 0).astype(float),
            parents=[],
            noise_dist=stats.norm(0, 1)
        ),
        'DrugDose': SCMVariable(
            name='DrugDose',
            equation=lambda u_d: np.abs(u_d),
            parents=[],
            noise_dist=stats.norm(1, 0.5)
        ),
        'DrugMetabolism': SCMVariable(
            name='DrugMetabolism',
            equation=lambda genotype, dose, u_m: genotype * 0.5 * dose + 0.1 * u_m,
            parents=['Genotype', 'DrugDose'],
            noise_dist=stats.norm(0, 1)
        ),
        'Response': SCMVariable(
            name='Response',
            equation=lambda dose, metabolism, u_r: 2*dose - metabolism + 0.2*u_r,
            parents=['DrugDose', 'DrugMetabolism'],
            noise_dist=stats.norm(0, 1)
        )
    }
    
    return StructuralCausalModel(variables)


def cell_cycle_confounding_scm() -> StructuralCausalModel:
    """
    Cell cycle as confounder in perturbation experiments.
    
    Structural equations:
        CellCycle = U_CellCycle (latent cell cycle phase)
        Transfection = sigmoid(CellCycle) + U_Transfection
        GeneExpression = 2*Transfection + CellCycle + U_Expression
    
    Cell cycle affects both transfection efficiency and gene expression.
    
    Returns
    -------
    StructuralCausalModel
        Cell cycle confounding SCM
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    variables = {
        'CellCycle': SCMVariable(
            name='CellCycle',
            equation=lambda u_cc: u_cc,
            parents=[],
            noise_dist=stats.norm(0, 1)
        ),
        'Transfection': SCMVariable(
            name='Transfection',
            equation=lambda cc, u_t: sigmoid(cc) + 0.1*u_t,
            parents=['CellCycle'],
            noise_dist=stats.norm(0, 1)
        ),
        'GeneExpression': SCMVariable(
            name='GeneExpression',
            equation=lambda transfection, cc, u_g: 2*transfection + 0.5*cc + 0.2*u_g,
            parents=['Transfection', 'CellCycle'],
            noise_dist=stats.norm(0, 1)
        )
    }
    
    return StructuralCausalModel(variables)


def collider_scm() -> StructuralCausalModel:
    """
    Collider structure: X -> C <- Y
    
    Structural equations:
        X = U_X
        Y = U_Y
        C = X + Y + U_C
    
    C is a collider. Conditioning on C creates spurious correlation between X and Y.
    
    Returns
    -------
    StructuralCausalModel
        Collider SCM
    """
    variables = {
        'X': SCMVariable(
            name='X',
            equation=lambda u_x: u_x,
            parents=[],
            noise_dist=stats.norm(0, 1)
        ),
        'Y': SCMVariable(
            name='Y',
            equation=lambda u_y: u_y,
            parents=[],
            noise_dist=stats.norm(0, 1)
        ),
        'C': SCMVariable(
            name='C',
            equation=lambda x, y, u_c: x + y + 0.1*u_c,
            parents=['X', 'Y'],
            noise_dist=stats.norm(0, 1)
        )
    }
    
    return StructuralCausalModel(variables)


def mediation_scm() -> StructuralCausalModel:
    """
    Mediation structure: X -> M -> Y, with direct effect X -> Y
    
    Structural equations:
        X = U_X
        M = 2*X + U_M (mediator)
        Y = 0.5*X + 1.5*M + U_Y (direct + indirect effect)
    
    Returns
    -------
    StructuralCausalModel
        Mediation SCM
    """
    variables = {
        'X': SCMVariable(
            name='X',
            equation=lambda u_x: u_x,
            parents=[],
            noise_dist=stats.norm(0, 1)
        ),
        'M': SCMVariable(
            name='M',
            equation=lambda x, u_m: 2*x + 0.2*u_m,
            parents=['X'],
            noise_dist=stats.norm(0, 1)
        ),
        'Y': SCMVariable(
            name='Y',
            equation=lambda x, m, u_y: 0.5*x + 1.5*m + 0.2*u_y,
            parents=['X', 'M'],
            noise_dist=stats.norm(0, 1)
        )
    }
    
    return StructuralCausalModel(variables)


# Linear SCM examples for efficient counterfactual computation

def linear_treatment_effect_scm() -> LinearSCM:
    """
    Linear SCM for treatment effect estimation.
    
    Structural equations:
        Z = U_Z (confounder)
        T = 0.5*Z + U_T (treatment)
        Y = 2*T + Z + U_Y (outcome)
    
    Returns
    -------
    LinearSCM
        Linear treatment effect SCM
    """
    coefficients = {
        'T': {'Z': 0.5},
        'Y': {'T': 2.0, 'Z': 1.0}
    }
    
    noise_distributions = {
        'Z': stats.norm(0, 1),
        'T': stats.norm(0, 0.5),
        'Y': stats.norm(0, 0.5)
    }
    
    return LinearSCM(coefficients, noise_distributions)


def linear_instrumental_variable_scm() -> LinearSCM:
    """
    Linear SCM with instrumental variable structure.
    
    Structural equations:
        Z = U_Z (instrument)
        X = Z + U_X (treatment, affected by instrument)
        Y = 2*X + U_Y (outcome)
    
    U_X and U_Y are correlated (unmeasured confounding).
    
    Returns
    -------
    LinearSCM
        Linear IV SCM
    """
    coefficients = {
        'X': {'Z': 1.0},
        'Y': {'X': 2.0}
    }
    
    noise_distributions = {
        'Z': stats.norm(0, 1),
        'X': stats.norm(0, 0.5),
        'Y': stats.norm(0, 0.5)
    }
    
    return LinearSCM(coefficients, noise_distributions)
