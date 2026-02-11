import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from globalemu.eval import evaluate

def main():
    base_dir = sys.argv[1]
    if base_dir[-1] != '/':
        base_dir += '/'
    
    config = np.genfromtxt(
        base_dir + 'gui_configuration.csv',
        delimiter=',',
        names=True,
        dtype='U100,f8,f8,f8,f8,U100,U100',  # Explicitly specify types
        encoding='utf-8'
    )
    
    logs = config['logs'].tolist()
    logs = [int(x) for x in logs if x != '--']
    label_min = config['label_min'][0]
    label_max = config['label_max'][0]
    ylabel = config['ylabel'][0]
    
    predictor = evaluate(base_dir=base_dir, logs=logs)
    
    # Calculate initial values (center of range)
    center = []
    for i in range(len(config['names'])):
        center.append(config['mins'][i] + (config['maxs'][i] - config['mins'][i])/2)
    center = np.array(center)
    
    # Create figure with plot on left, sliders on right
    n_sliders = len(config['names'])
    fig = plt.figure(figsize=(14, 8))
    
    # Main plot on left side - takes full height
    ax_plot = plt.subplot2grid((n_sliders, 2), (0, 0), rowspan=n_sliders)
    
    # Initial signal
    def get_params(slider_vals):
        params = []
        for i in range(len(slider_vals)):
            if i in set(logs):
                params.append(10**slider_vals[i])
            else:
                params.append(slider_vals[i])
        return params
    
    signal, z = predictor(get_params(center))
    line, = ax_plot.plot(z, signal, c='k', lw=2)
    ax_plot.set_xlabel('z')
    ax_plot.set_ylabel(ylabel)
    ax_plot.set_ylim([label_min, label_max])
    ax_plot.grid(True, alpha=0.3)
    
    # Create sliders on right side
    sliders = []
    
    for i in range(n_sliders):
        # Create axis for this slider on right column
        ax_slider = plt.subplot2grid((n_sliders, 2), (i, 1))
        
        # Create slider
        slider = Slider(
            ax_slider,
            config['names'][i],
            config['mins'][i],
            config['maxs'][i],
            valinit=center[i],
            valstep=(config['maxs'][i] - config['mins'][i])/100
        )
        sliders.append(slider)
    
    # Update function
    def update(val):
        slider_vals = [s.val for s in sliders]
        params = get_params(slider_vals)
        signal, z = predictor(params)
        line.set_ydata(signal)
        fig.canvas.draw_idle()
    
    # Connect all sliders to update function
    for slider in sliders:
        slider.on_changed(update)
    
    # Reset button
    ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
    btn_reset = Button(ax_reset, 'Reset')
    
    def reset(event):
        for i, slider in enumerate(sliders):
            slider.set_val(center[i])
    
    btn_reset.on_clicked(reset)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()