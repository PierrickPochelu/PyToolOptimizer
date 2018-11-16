<h1>PyToolOptimizer</h1>

This project is a tool to help searchers in Optimization to code exciting new optimizer ! 

You can use any part or all this project to help you to implement new optimizer or new function to minimize. It's especially handsome in Deep Learning field because our tool abstracts complexity of the Deep Learning framework Tensorflow.

<h2>Why this framework is usefull ?</h2>
The use of this framework is particularly recommended in the field of Deep Learning where update weight directly in TF is difficult to implement. This framework proposes to abstract this complexity with the forward and gradient descent are computed in GPU and the optimization is partially written in Python.

<h2>How to use it ?</h2>
The usage is very simple :
```
function_to_min = Rosenbrock() <br/>
optimizer = MCMC() <br/>
x=np.random.uniform(-1,+1,(2,)) <br/>
x_n=optimizer.run_on_step(x,function_to_min) <br/>
```
x_n contains the new position. <br/>

<p>
More examples here :
<ul>
  <li> <b>simple.py</b> to a complete example</li>
  <li> <b>bench.py test</b> many strategies as Deep Learning optimizer </li>
</ul>
</p>
