import numpy as np
import bokeh
from bokeh.models import Circle, ColumnDataSource, Line, LinearAxis, Range1d
from bokeh.plotting import figure, output_notebook, show
from bokeh.core.properties import value

# Load validation records
control_validation = np.load("control_validation.npy", allow_pickle=True)
delayed_validation = np.load("delayed_model_validation.npy", allow_pickle=True)
delayed_validation_2 = np.load("delayed_model_validation_lr_shrink_downscale.npy", allow_pickle=True)


control_validation_data = {"x": [], "y": []}
for epoch_data in control_validation:
    control_validation_data["x"].extend([x["Epoch"] + (x["Batch"] / 1000) for x in epoch_data])
    control_validation_data["y"].extend([x["Accuracy"] for x in epoch_data])
    

delayed_validation_data = {"x": [], "y": []}
for epoch_data in delayed_validation:
    delayed_validation_data["x"].extend([x["Epoch"] + (x["Batch"] / 1000) for x in epoch_data])
    delayed_validation_data["y"].extend([x["Accuracy"] for x in epoch_data])
    

delayed_validation_data_2 = {"x": [], "y": []}
for epoch_data in delayed_validation_2:
    delayed_validation_data_2["x"].extend([x["Epoch"] + (x["Batch"] / 1000) for x in epoch_data])
    delayed_validation_data_2["y"].extend([x["Accuracy"] for x in epoch_data])
    



p = figure()
p.line("x", "y", source=control_validation_data)
p.line("x", "y", color="red", source=delayed_validation_data)
p.line("x", "y", color="green", source=delayed_validation_data_2)
show(p)

control_test = np.load("control_test.npy", allow_pickle=True)
delayed_test = np.load("delayed_model_test.npy", allow_pickle=True)
delayed_test_2 = np.load("delayed_model_test_lr_shrink_downscale.npy", allow_pickle=True)


control_test_data = {
    "x": [x["Epoch"] for x in control_test], 
    "y": [x["Accuracy"] for x in control_test]
}
delayed_test_data = {
    "x": [x["Epoch"] for x in delayed_test], 
    "y": [x["Accuracy"] for x in delayed_test]
}
delayed_test_data_2 = {
    "x": [x["Epoch"] for x in delayed_test_2], 
    "y": [x["Accuracy"] for x in delayed_test_2]
}

p2 = figure(title="Model Test Accuracy over Epochs of Training", x_axis_label="Training Epochs",)
p2.line("x", "y", source = control_test_data, legend="Standard Model")
p2.line("x", "y", color="red", source = delayed_test_data, legend="Delayed Model")
p2.line("x", "y", color="green", source = delayed_test_data_2, legend="Delayed Model (Delayed Gradients Downscaled)")
p2.yaxis[0].axis_label = "Accuracy"
p2.legend.location = "bottom_right"
show(p2)
