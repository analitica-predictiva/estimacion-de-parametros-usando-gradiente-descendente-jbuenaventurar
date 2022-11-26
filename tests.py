"""
Calificación del laboratorio
-----------------------------------------------------------------------------------------
"""

import sys

import numpy as np

import preguntas



def test_01():
    """
    ---< Run command >-------------------------------------------------------------------
    Pregunta 01
    pip3 install scikit-learn pandas numpy
    python3 tests.py 01
    """

    x_poly, _ = preguntas.pregunta_01()
    x_poly = x_poly.round(3)
    x_expected = np.array(
        [
            [1.0, -4.0, 16.0],
            [1.0, -3.579, 12.809],
            [1.0, -3.158, 9.972],
            [1.0, -2.737, 7.49],
            [1.0, -2.316, 5.363],
            [1.0, -1.895, 3.59],
            [1.0, -1.474, 2.172],
            [1.0, -1.053, 1.108],
            [1.0, -0.632, 0.399],
            [1.0, -0.21, 0.044],
            [1.0, 0.21, 0.044],
            [1.0, 0.632, 0.399],
            [1.0, 1.053, 1.108],
            [1.0, 1.474, 2.172],
            [1.0, 1.895, 3.59],
            [1.0, 2.316, 5.363],
            [1.0, 2.737, 7.49],
            [1.0, 3.158, 9.972],
            [1.0, 3.579, 12.809],
            [1.0, 4.0, 16.0],
        ]
    )

    for i in range(x_poly.shape[0]):
        for j in range(x_poly.shape[1]):
            assert x_poly[i, j] == x_expected[i, j]


def pregunta_02(): 
 
    # Importe numpy 
    import numpy as np 
 
    x_poly, y = pregunta_01() 
 
    # Fije la tasa de aprendizaje en 0.0001 y el número de iteraciones en 1000 
    learning_rate = 0.0001 
    n_iterations = 1000 
 
    # Defina el parámetro inicial params como un arreglo de tamaño 3 con ceros 
    intercept_ = np.mean(np.array(y)) 
    params = np.zeros(x_poly.shape[1]) 
    for epochs in range(n_iterations): 
 
        # Compute el pronóstico con los parámetros actuales 
        y_pred = np.dot(x_poly, params) 
 
        # Calcule el error 
        error = (y - y_pred) / 2 
 
        # Calcule el gradiente 
        gradient = -2*np.sum(np.multiply(x_poly,np.array(error)[:,np.newaxis]),axis=0) 
        #gradient =  -2*sum(error) 
 
        # Actualice los parámetros 
        params = params - learning_rate * gradient 
 
    return params
