# wbd run notes

the wbd contains GANBLR models proposed by `Tulip Lab` for tabular data generation, which can sample fully artificial data from real data.

Currently, this package contains following GANBLR components:

- kdb encoder
- broad component

The kdb encoder "KdbHighOrderFeatureEncoder", it is used to build the embedding for broad and wide component in wbd. 

# Install

We recommend you to sync ganblr through pip:

```bash
pip install ganblr
```

Alternatively, you can also clone the repository and install it from sources.

```bash
git clone git@github.com:tulip-lab/ganblr.git
cd ganblr
python setup.py install
```

Also, the tensorflow is best on 2.8 with Keras on 2.8 as well. 
If you face any issues of running the code (particularly on notebook version), please double check your version with tensorflow and keras. It is best to run under docker environment.

# Usage Example

In this example we load the `adult-dm.csv` and run the `wbd_run_notebook.ipynb`. 
Make sure the data used here has been discretized already and the input must be int32 with integer type. 

Run the below `wbdf` function to define the model. 
```
def wbdf()
```
if you want to add more component with wbd, you could add the component before the code below and insert the component result into the concat
```
output = tf.concat([widebroad_output, dnn_output], axis=-1)
```

