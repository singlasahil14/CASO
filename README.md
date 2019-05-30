Context Aware Second-Order Interpretation
=====================================

Code for reproducing experiments in ["Understanding Impacts of High-Order Loss Approximations and Features in Deep Learning Interpretation"](https://arxiv.org/abs/1902.00407).


## Prerequisites

- Python, NumPy, Pytorch, Argparse, Matplotlib
- A recent NVIDIA GPU

## Basic Usage

To evaluate the interpretation with default parameters on the given toy image, run python main.py. To access all the parameters use python main.py --help.

## Examples

<p>To generate the following examples use python main.py --lambda1 LAMBDA</p>

<div align = 'center'>
	<figure>
	    <div style="float:left" >
			<img src = 'examples/duck.jpeg' height = '200px' width = '200px'>
	  		<figcaption>Image</figcaption>
	    </div>
	    <div style="float:left" >
			<img src = 'examples/delta_5e-4.png' height = '200px' width = '200px'>
	  		<figcaption>&#955;<sub>1</sub>=5e-4</figcaption>
	    </div>
	    <div style="float:left" >
			<img src = 'examples/delta_1e-4.png' height = '200px' width = '200px'>
			<figcaption>&#955;<sub>1</sub>=1e-4</figcaption>
	    </div>
	    <div style="float:left" >
			<img src = 'examples/delta_1e-5.png' height = '200px' width = '200px'>
			<figcaption>&#955;<sub>1</sub>=1e-5</figcaption>
	    </div>
	<br>
	<br>
	<br>
	<br>
	<br>
	<br>
	<br>
	<br>
	</figure>
	<br>
</div>
<br>

To generate the CAFO example, use python .\main.py --image_path=IMAGE_NAME --lambda1=0 --magnitude <br>
For the CASO example, use python .\main.py --image_path==IMAGE_NAME --lambda1=0 --magnitude --second-order

<div align = 'center'>
	<figure>
	    <div style="float:left" >
			<img src = 'examples/turtle.jpeg' height = '200px' width = '200px'>
	  		<figcaption>Confidence=0.213</figcaption>
	    </div>
	    <div style="float:left" >
			<img src = 'examples/turtle_cafo.png' height = '200px' width = '200px'>
			<figcaption>CAFO output</figcaption>
	    </div>
	    <div style="float:left" >
			<img src = 'examples/turtle_caso.png' height = '200px' width = '200px'>
			<figcaption>CASO output</figcaption>
	    </div>
	<br>
	<br>
	<br>
	<br>
	<br>
	<br>
	<br>
	<br>
	</figure>
	<br>
</div>


<div align = 'center'>
	<figure>
	    <div style="float:left" >
			<img src = 'examples/duck.jpeg' height = '200px' width = '200px'>
	  		<figcaption>Confidence=0.957</figcaption>
	    </div>
	    <div style="float:left" >
			<img src = 'examples/duck_cafo.png' height = '200px' width = '200px'>
			<figcaption>CAFO output</figcaption>
	    </div>
	    <div style="float:left" >
			<img src = 'examples/duck_caso.png' height = '200px' width = '200px'>
			<figcaption>CASO output</figcaption>
	    </div>
	</figure>
	<br>
</div>
