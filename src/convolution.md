# Convolutional Neural Network

## Useful links

https://www.youtube.com/watch?v=Lakz2MoHy6o

## The math

$$
  \text{Terms:}
  \ \\
  \ \\

  * = \text{convolution} \\
  \star = \text{cross correlation} \\
  B = \text{bias} \\
  X = \text{activation in previous layer} \\
  Y = \text{activation in current layer} \\
  d = \text{number of nodes in current layer} \\
  K = \text{kernel matrix with weights} \\
  j = \text{index into previous layer} \\
  i = \text{index into current layer} \\

  \ \\
  \ \\

  \text{Convolutional layer:}

  \ \\
  \ \\

  Y_1=B_1+X_1 \star K_{11} + ...\ + X_n \star K_{1n}\\
  Y_2=B_2+X_1 \star K_{21} + ...\ + X_n \star K_{2n}\\
  \vdots \\
  Y_2=B_d+X_1 \star K_{d1} + ...\ + X_n \star K_{dn}\\

  \ \\
  \ \\

  \text{Forward propagation:}

  \ \\
  \ \\

  Y_i=
  B_i+
  \sum_{j=1}^n
  X_j
  \star K_{ij}
  \\
  i=1 ... d
$$
