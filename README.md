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
	<figure style='float: left; margin-left: 5px; margin-right: 5px'>
		<img src = 'examples/duck.jpeg' width = '212px'>
	  	<figcaption>Image</figcaption>
	</figure>
	<figure style='float: left; margin-left: 5px; margin-right: 5px'>
		<img src = 'examples/delta_5e-4.png' width = '212px'>
	  	<figcaption>&#955;<sub>1</sub>=5e-4</figcaption>
	</figure>
	<figure style='float: left; margin-left: 5px; margin-right: 5px'>
		<img src = 'examples/delta_1e-4.png' width = '212px'>
		<figcaption>&#955;<sub>1</sub>=1e-4</figcaption>
	</figure>
	<figure style='float: left; margin-left: 5px; margin-right: 5px'>
		<img src = 'examples/delta_1e-5.png' width = '212px'>
		<figcaption>&#955;<sub>1</sub>=1e-5</figcaption>
	</figure>
</div>

To generate the CAFO example, use python .\main.py --image_path=IMAGE_NAME --lambda1=0 --magnitude <br>
For the CASO example, use python .\main.py --image_path==IMAGE_NAME --lambda1=0 --magnitude --second-order

<div align = 'center'>
	    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
			<img src = 'examples/turtle.jpeg' width = '240px'>
	  		<figcaption>Confidence=0.213</figcaption>
	    </figure>
	    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
			<img src = 'examples/turtle_cafo.png' width = '240px'>
			<figcaption>CAFO output</figcaption>
	    </figure>
	    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
			<img src = 'examples/turtle_caso.png' width = '240px'>
			<figcaption>CASO output</figcaption>
	    </figure>
</div>

<div align = 'center'>
	    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
			<img src = 'examples/duck.jpeg' width = '240px'>
	  		<figcaption>Confidence=0.957</figcaption>
	    </figure>
	    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
			<img src = 'examples/duck_cafo.png' width = '240px'>
			<figcaption>CAFO output</figcaption>
	    </figure>
	    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
			<img src = 'examples/duck_caso.png' width = '240px'>
			<figcaption>CASO output</figcaption>
	    </figure>
</div>

<p>My <img src="examples/duck.jpeg" alt="foo bar" title="CASO output" /></p>
