Context Aware Second-Order Interpretation
=====================================

Code for reproducing experiments in ["Understanding Impacts of High-Order Loss Approximations and Features in Deep Learning Interpretation"](https://arxiv.org/abs/1902.00407).


## Prerequisites

- Python, NumPy, Pytorch, Argparse, Matplotlib
- A recent NVIDIA GPU

## Basic Usage

To evaluate the interpretation with default parameters on the given toy image, run python main.py. To access all the parameters use python main.py --help.

<style>
.site-main table.no-border td {
    border: none;
}
</style>

## Examples

<p>To generate the following examples use python main.py --lambda1 LAMBDA</p>

<div align = 'center'>
	<figure style='float: left; margin-left: 5px; margin-right: 5px'>
		<img src = 'examples/duck.jpeg' width = '212px'>
	  	<p align="center">Image</p>
	</figure>
	<figure style='float: left; margin-left: 5px; margin-right: 5px'>
		<img src = 'examples/delta_5e-4.png' width = '212px'>
	  	<p align="center">&#955;<sub>1</sub>=5e-4</p>
	</figure>
	<figure style='float: left; margin-left: 5px; margin-right: 5px'>
		<img src = 'examples/delta_1e-4.png' width = '212px'>
		<p align="center">&#955;<sub>1</sub>=1e-4</p>
	</figure>
	<figure style='float: left; margin-left: 5px; margin-right: 5px'>
		<img src = 'examples/delta_1e-5.png' width = '212px'>
		<p align="center">&#955;<sub>1</sub>=1e-5</p>
	</figure>
</div>

To generate the CAFO example, use python .\main.py --image_path=IMAGE_NAME --lambda1=0 --magnitude <br>
For the CASO example, use python .\main.py --image_path==IMAGE_NAME --lambda1=0 --magnitude --second-order

<div align = 'center'>
	    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
			<img src = 'examples/turtle.jpeg' width = '240px'>
	  		<p align="center">Confidence=0.213</p>
	    </figure>
	    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
			<img src = 'examples/turtle_cafo.png' width = '240px'>
			<p align="center">CAFO output</p>
	    </figure>
	    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
			<img src = 'examples/turtle_caso.png' width = '240px'>
			<p align="center">CASO output</p>
	    </figure>
</div>

<div align = 'center'>
	    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
			<img src = 'examples/duck.jpeg' width = '240px'>
	  		<p>Confidence=0.957</p>
	    </figure>
	    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
			<img src = 'examples/duck_cafo.png' width = '240px'>
			<p>CAFO output</p>
	    </figure>
	    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
			<img src = 'examples/duck_caso.png' width = '240px'>
			<p>CASO output</p>
	    </figure>
</div>

<table width="500px" border="0px" cellspacing="0px" cellpadding="0px" class="no-border">
<tr>
<td align="center" valign="center">
	<figure style='float: left; margin-left: 5px; margin-right: 5px'>
	<img src = 'examples/duck.jpeg' width = '240px'>
	<p>Confidence=0.957</p>
	</figure>
</td>

<td align="center" valign="center">
    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
		<img src = 'examples/duck_cafo.png' width = '240px'>
		<p>CAFO output</p>
    </figure>
</td>

<td align="center" valign="center">
    <figure style='float: left; margin-left: 5px; margin-right: 5px'>
		<img src = 'examples/duck_caso.png' width = '240px'>
		<p>CASO output</p>
    </figure>
</td>

</tr>
</table>
