def plot_parameter_traces(parameter_traces, parameter_names, save_path=None):
    import matplotlib.pyplot as plt

    num_params = len(parameter_names)
    fig, axes = plt.subplots(num_params, 1, figsize=(10, 2 * num_params))

    for i, param_name in enumerate(parameter_names):
        axes[i].plot(parameter_traces[param_name])
        axes[i].set_title(f'Trace of {param_name}')
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel(param_name)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()