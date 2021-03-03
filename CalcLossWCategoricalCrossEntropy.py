# e ** x = b
# solving for x

# import numpy as np
# b = 5.2
# print(np.log(b))
# print(np.e ** np.log(b))

import math

softmax_output = [.7, .1, .2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])
print(loss)
