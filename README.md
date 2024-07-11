# GradientFlows

[![Build Status](https://github.com/vilin97/GradientFlows.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/vilin97/GradientFlows.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/vilin97/GradientFlows.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/vilin97/GradientFlows.jl)

Code for [Transport based particle methods for the Fokker-Planck-Landau equation](https://arxiv.org/abs/2405.10392).

We propose a particle method for numerically solving the Landau equation, inspired by the score-based transport modeling (SBTM) method for the Fokker-Planck equation. This method can preserve some important physical properties of the Landau equation, such as the conservation of mass, momentum, and energy, and decay of estimated entropy. We prove that matching the gradient of the logarithm of the approximate solution is enough to recover the true solution to the Landau equation with Maxwellian molecules. Several numerical experiments in low and moderately high dimensions are performed, with particular emphasis on comparing the proposed method with the traditional particle or blob method.
