import tensorflow as tf
from rich import print as rprint

a = tf.constant([5.0, 3.0])
b = tf.constant([2.0, 4.0])
c = a + b
rprint('eager Execution : ', c.numpy())

@tf.function
def multiply_result(a,b):
    return a * b

result = multiply_result(a,b)
rprint('tf.function Execution : ', result.numpy())